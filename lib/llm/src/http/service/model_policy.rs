// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend model-selection policies.
//!
//! A policy maps a client-facing virtual model name to a concrete model already
//! registered with Dynamo. Worker selection remains the responsibility of the
//! existing per-model router after this policy runs.

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, bail};
use rand::Rng;
use serde::Deserialize;

use super::stage_router::{PickerMode, Tier, select_tier};
use dynamo_protocols::types::ChatCompletionRequestMessage;

#[derive(Debug, Deserialize)]
struct PolicyFile {
    schema_version: u32,
    targets: HashMap<String, Target>,
    routes: HashMap<String, Route>,
}

#[derive(Debug, Deserialize)]
struct Target {
    /// Concrete Dynamo model display name.
    id: String,
}

#[derive(Debug, Deserialize)]
struct Route {
    /// Client-facing virtual model name. Defaults to the TOML table key.
    id: Option<String>,
    #[serde(flatten)]
    kind: RouteKind,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RouteKind {
    Passthrough {
        target: String,
    },
    Random {
        targets: Vec<String>,
        #[serde(default)]
        weights: Option<Vec<f64>>,
    },
    StageRouter {
        capable_target: String,
        efficient_target: String,
        picker: PickerMode,
        confidence_threshold: f64,
        #[serde(default = "default_recent_turn_window")]
        recent_turn_window: usize,
    },
}

fn default_recent_turn_window() -> usize {
    3
}

#[derive(Clone, Debug)]
enum CompiledRoute {
    Passthrough {
        model: String,
    },
    Random {
        models: Vec<String>,
        cumulative_weights: Vec<f64>,
        total_weight: f64,
    },
    StageRouter {
        capable_model: String,
        efficient_model: String,
        picker: PickerMode,
        confidence_threshold: f64,
        recent_turn_window: usize,
    },
}

/// Immutable, startup-validated model-selection policy.
#[derive(Clone, Debug, Default)]
pub struct ModelPolicy {
    routes: HashMap<String, CompiledRoute>,
}

impl ModelPolicy {
    pub fn from_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read model-router config {}", path.display()))?;
        Self::from_toml(&text)
            .with_context(|| format!("invalid model-router config {}", path.display()))
    }

    fn from_toml(text: &str) -> Result<Self> {
        let file: PolicyFile = toml::from_str(text).context("failed to parse TOML")?;
        let PolicyFile {
            schema_version,
            targets,
            routes: configured_routes,
        } = file;
        if schema_version != 1 {
            bail!("unsupported schema_version {}; expected 1", schema_version);
        }

        let resolve_target = |name: &str| -> Result<String> {
            let target = targets
                .get(name)
                .with_context(|| format!("route references unknown target '{name}'"))?;
            if target.id.trim().is_empty() {
                bail!("target '{name}' has an empty id");
            }
            Ok(target.id.clone())
        };

        let mut routes = HashMap::with_capacity(configured_routes.len());
        for (table_name, route) in configured_routes {
            let route_id = route.id.unwrap_or(table_name);
            if route_id.trim().is_empty() {
                bail!("route id cannot be empty");
            }

            let compiled = match route.kind {
                RouteKind::Passthrough { target } => CompiledRoute::Passthrough {
                    model: resolve_target(&target)?,
                },
                RouteKind::Random { targets, weights } => {
                    if targets.is_empty() {
                        bail!("random route '{route_id}' must contain at least one target");
                    }
                    let weights = weights.unwrap_or_else(|| vec![1.0; targets.len()]);
                    if weights.len() != targets.len() {
                        bail!(
                            "random route '{route_id}' has {} targets but {} weights",
                            targets.len(),
                            weights.len()
                        );
                    }

                    let mut total_weight = 0.0;
                    let mut cumulative_weights = Vec::with_capacity(weights.len());
                    for weight in weights {
                        if !weight.is_finite() || weight <= 0.0 {
                            bail!(
                                "random route '{route_id}' weights must be finite and greater than zero"
                            );
                        }
                        total_weight += weight;
                        cumulative_weights.push(total_weight);
                    }
                    let models = targets
                        .iter()
                        .map(|target| resolve_target(target))
                        .collect::<Result<Vec<_>>>()?;
                    CompiledRoute::Random {
                        models,
                        cumulative_weights,
                        total_weight,
                    }
                }
                RouteKind::StageRouter {
                    capable_target,
                    efficient_target,
                    picker,
                    confidence_threshold,
                    recent_turn_window,
                } => {
                    if !confidence_threshold.is_finite()
                        || !(0.0..=1.0).contains(&confidence_threshold)
                    {
                        bail!(
                            "stage route '{route_id}' confidence_threshold must be between 0 and 1"
                        );
                    }
                    if recent_turn_window == 0 {
                        bail!("stage route '{route_id}' recent_turn_window must be at least 1");
                    }
                    CompiledRoute::StageRouter {
                        capable_model: resolve_target(&capable_target)?,
                        efficient_model: resolve_target(&efficient_target)?,
                        picker,
                        confidence_threshold,
                        recent_turn_window,
                    }
                }
            };

            if routes.insert(route_id.clone(), compiled).is_some() {
                bail!("duplicate route id '{route_id}'");
            }
        }

        Ok(Self { routes })
    }

    /// Resolve a virtual model to a concrete Dynamo model. Concrete model names
    /// that are not routes pass through unchanged.
    pub fn resolve(&self, requested_model: &str) -> String {
        self.resolve_with_sample(requested_model, rand::rng().random::<f64>())
    }

    fn resolve_with_sample(&self, requested_model: &str, sample: f64) -> String {
        match self.routes.get(requested_model) {
            None => requested_model.to_string(),
            Some(CompiledRoute::Passthrough { model }) => model.clone(),
            Some(CompiledRoute::Random {
                models,
                cumulative_weights,
                total_weight,
            }) => {
                let draw = sample.clamp(0.0, 1.0 - f64::EPSILON) * total_weight;
                let index = cumulative_weights.partition_point(|weight| *weight <= draw);
                models[index.min(models.len() - 1)].clone()
            }
            Some(CompiledRoute::StageRouter {
                capable_model,
                efficient_model,
                picker,
                ..
            }) => match picker {
                PickerMode::CapableFirst => capable_model.clone(),
                PickerMode::EfficientFirst => efficient_model.clone(),
            },
        }
    }

    /// Resolve a Chat Completions request, including semantic policies that
    /// inspect coding-agent tool history.
    pub fn resolve_chat(
        &self,
        requested_model: &str,
        messages: &[ChatCompletionRequestMessage],
    ) -> String {
        let Some(CompiledRoute::StageRouter {
            capable_model,
            efficient_model,
            picker,
            confidence_threshold,
            recent_turn_window,
        }) = self.routes.get(requested_model)
        else {
            return self.resolve(requested_model);
        };

        let decision = select_tier(
            messages,
            *picker,
            *confidence_threshold,
            *recent_turn_window,
        );
        tracing::info!(
            route = requested_model,
            source = ?decision.source,
            confidence = ?decision.confidence,
            tier = ?decision.tier,
            "Dynamo stage router selected model tier"
        );
        match decision.tier {
            Tier::Capable => capable_model.clone(),
            Tier::Efficient => efficient_model.clone(),
        }
    }

    pub fn route_ids(&self) -> impl Iterator<Item = &str> {
        self.routes.keys().map(String::as_str)
    }

    pub fn contains_route(&self, model: &str) -> bool {
        self.routes.contains_key(model)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelPolicy;
    use dynamo_protocols::types::ChatCompletionRequestMessage;

    const CONFIG: &str = r#"
schema_version = 1

[targets.weak]
id = "Qwen/Qwen3-0.6B"

[targets.strong]
id = "Qwen/Qwen3-8B"

[routes.fast]
id = "fast"
type = "passthrough"
target = "weak"

[routes.smart]
id = "smart"
type = "random"
targets = ["weak", "strong"]
weights = [3.0, 1.0]

[routes.stage]
id = "stage"
type = "stage_router"
capable_target = "strong"
efficient_target = "weak"
picker = "efficient_first"
confidence_threshold = 0.5
"#;

    #[test]
    fn passthrough_route_resolves_concrete_model() {
        let policy = ModelPolicy::from_toml(CONFIG).unwrap();
        assert_eq!(policy.resolve_with_sample("fast", 0.9), "Qwen/Qwen3-0.6B");
        assert_eq!(
            policy.resolve_with_sample("already-concrete", 0.2),
            "already-concrete"
        );
    }

    #[test]
    fn weighted_random_route_obeys_boundaries() {
        let policy = ModelPolicy::from_toml(CONFIG).unwrap();
        assert_eq!(policy.resolve_with_sample("smart", 0.0), "Qwen/Qwen3-0.6B");
        assert_eq!(
            policy.resolve_with_sample("smart", 0.749),
            "Qwen/Qwen3-0.6B"
        );
        assert_eq!(policy.resolve_with_sample("smart", 0.75), "Qwen/Qwen3-8B");
        assert_eq!(policy.resolve_with_sample("smart", 0.999), "Qwen/Qwen3-8B");
    }

    #[test]
    fn rejects_invalid_random_weights() {
        let config = CONFIG.replace("weights = [3.0, 1.0]", "weights = [1.0]");
        let error = ModelPolicy::from_toml(&config).unwrap_err().to_string();
        assert!(error.contains("2 targets but 1 weights"));
    }

    #[test]
    fn exposes_virtual_model_ids() {
        let policy = ModelPolicy::from_toml(CONFIG).unwrap();
        assert!(policy.contains_route("fast"));
        assert!(policy.contains_route("smart"));
        assert_eq!(policy.route_ids().count(), 3);
    }

    #[test]
    fn stage_route_uses_picker_default_without_tool_history() {
        let policy = ModelPolicy::from_toml(CONFIG).unwrap();
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_str(r#"[{"role":"user","content":"implement the requested change"}]"#)
                .unwrap();
        assert_eq!(policy.resolve_chat("stage", &messages), "Qwen/Qwen3-0.6B");
    }
}
