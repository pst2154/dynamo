// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Signal-driven model-tier selection for coding-agent traffic.

use std::collections::HashMap;

use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessageContent,
};
use serde::Deserialize;

const STALL_MIN_TURN_DEPTH: u32 = 8;
const SCORE_GAIN: f64 = 5.0;
const HARD_SEVERITY: f64 = 0.7;
const SIGNAL_UNIT: f64 = 0.10;

const EDIT_TOOLS: &[&str] = &[
    "edit",
    "multiedit",
    "notebookedit",
    "str_replace",
    "str_replace_based_edit_tool",
    "apply_patch",
    "text_editor",
    "patch",
];
const WRITE_TOOLS: &[&str] = &["write", "create_file", "new_file", "write_file"];
const READ_TOOLS: &[&str] = &["read", "view", "read_file", "search_files"];
const PLAN_TOOLS: &[&str] = &["todowrite", "todo_write", "todo", "update_plan"];
const SHELL_TOOLS: &[&str] = &[
    "bash",
    "shell_command",
    "shell",
    "local_shell_call",
    "terminal",
    "exec_command",
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PickerMode {
    CapableFirst,
    EfficientFirst,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    Capable,
    Efficient,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionSource {
    Override,
    TestsPassed,
    Dimensions,
    FallOpen,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageDecision {
    pub tier: Tier,
    pub source: DecisionSource,
    pub confidence: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Signals {
    severity: f64,
    recent_edit_count: u32,
    recent_write_count: u32,
    recent_read_count: u32,
    recent_plan_count: u32,
    tests_passed: bool,
    turn_depth: u32,
    compacted: bool,
    has_tool_activity: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolCategory {
    Write,
    Edit,
    Read,
    Plan,
    Other,
}

#[derive(Debug)]
struct ToolResult {
    category: ToolCategory,
    text: String,
}

pub fn select_tier(
    messages: &[ChatCompletionRequestMessage],
    picker: PickerMode,
    confidence_threshold: f64,
    recent_window: usize,
) -> StageDecision {
    let signals = extract_signals(messages, recent_window);
    let default_tier = match picker {
        PickerMode::CapableFirst => Tier::Capable,
        PickerMode::EfficientFirst => Tier::Efficient,
    };

    if signals.compacted || signals.severity >= 1.0 {
        return StageDecision {
            tier: Tier::Capable,
            source: DecisionSource::Override,
            confidence: Some(1.0),
        };
    }
    if signals.tests_passed
        && signals.recent_write_count + signals.recent_edit_count >= 1
        && signals.severity == 0.0
    {
        return StageDecision {
            tier: Tier::Efficient,
            source: DecisionSource::TestsPassed,
            confidence: None,
        };
    }
    if !signals.has_tool_activity {
        return StageDecision {
            tier: default_tier,
            source: DecisionSource::FallOpen,
            confidence: None,
        };
    }

    let recent_ops = signals.recent_write_count
        + signals.recent_edit_count
        + signals.recent_read_count
        + signals.recent_plan_count;
    let no_production = signals.recent_write_count == 0 && signals.recent_edit_count == 0;
    let investigating = signals.recent_read_count >= 1 || signals.recent_plan_count >= 1;
    let deep_enough = signals.turn_depth >= STALL_MIN_TURN_DEPTH;
    let spinning = if deep_enough && no_production && !investigating {
        1.0
    } else {
        0.0
    };
    let exploring = if deep_enough && no_production && investigating {
        1.0
    } else {
        0.0
    };
    let production_intensity = if recent_ops == 0 {
        0.0
    } else {
        f64::from(signals.recent_write_count + signals.recent_edit_count) / f64::from(recent_ops)
    };
    let raw = SIGNAL_UNIT
        * (signals.severity / HARD_SEVERITY + spinning + exploring - production_intensity);
    let score = (SCORE_GAIN * raw).tanh();
    let confidence = score.abs();

    if confidence >= confidence_threshold {
        StageDecision {
            tier: if score > 0.0 {
                Tier::Capable
            } else {
                Tier::Efficient
            },
            source: DecisionSource::Dimensions,
            confidence: Some(confidence),
        }
    } else {
        StageDecision {
            tier: default_tier,
            source: DecisionSource::FallOpen,
            confidence: Some(confidence),
        }
    }
}

fn extract_signals(messages: &[ChatCompletionRequestMessage], recent_window: usize) -> Signals {
    let mut calls = HashMap::new();
    let mut results = Vec::new();
    let mut compacted = false;

    for message in messages {
        let serialized = serde_json::to_string(message).unwrap_or_default();
        compacted |= serialized
            .to_ascii_lowercase()
            .contains("session is being continued");

        match message {
            ChatCompletionRequestMessage::Assistant(assistant) => {
                for call in assistant.tool_calls.iter().flatten() {
                    calls.insert(
                        call.id.clone(),
                        classify_tool(&call.function.name, &call.function.arguments),
                    );
                }
            }
            ChatCompletionRequestMessage::Tool(tool) => {
                let text = match &tool.content {
                    ChatCompletionRequestToolMessageContent::Text(text) => text.clone(),
                    ChatCompletionRequestToolMessageContent::Array(parts) => parts
                        .iter()
                        .map(|part| match part {
                            dynamo_protocols::types::ChatCompletionRequestToolMessageContentPart::Text(text) => text.text.as_str(),
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                results.push(ToolResult {
                    category: calls
                        .get(&tool.tool_call_id)
                        .copied()
                        .unwrap_or(ToolCategory::Other),
                    text,
                });
            }
            _ => {}
        }
    }

    let recent = results.iter().rev().take(recent_window).collect::<Vec<_>>();
    let mut signals = Signals {
        turn_depth: messages.len() as u32,
        compacted,
        has_tool_activity: !results.is_empty() || !calls.is_empty(),
        ..Signals::default()
    };
    for result in recent {
        match result.category {
            ToolCategory::Write => signals.recent_write_count += 1,
            ToolCategory::Edit => signals.recent_edit_count += 1,
            ToolCategory::Read => signals.recent_read_count += 1,
            ToolCategory::Plan => signals.recent_plan_count += 1,
            ToolCategory::Other => {}
        }
        signals.severity = signals.severity.max(error_severity(&result.text));
        signals.tests_passed |= test_passed(&result.text);
    }
    signals
}

fn classify_tool(name: &str, arguments: &str) -> ToolCategory {
    let name = name.to_ascii_lowercase();
    if WRITE_TOOLS.contains(&name.as_str()) {
        return ToolCategory::Write;
    }
    if EDIT_TOOLS.contains(&name.as_str()) {
        return ToolCategory::Edit;
    }
    if READ_TOOLS.contains(&name.as_str()) {
        return ToolCategory::Read;
    }
    if PLAN_TOOLS.contains(&name.as_str()) {
        return ToolCategory::Plan;
    }
    if SHELL_TOOLS.contains(&name.as_str()) {
        let command = serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|value| {
                ["command", "cmd", "input"]
                    .into_iter()
                    .find_map(|key| value.get(key)?.as_str().map(str::to_string))
            })
            .unwrap_or_else(|| arguments.to_string())
            .to_ascii_lowercase();
        if ["cat >", "cat >>", "echo >", "tee ", "write_text("]
            .iter()
            .any(|pattern| command.contains(pattern))
        {
            return ToolCategory::Write;
        }
        if ["sed -i", "patch ", "perl -i"]
            .iter()
            .any(|pattern| command.contains(pattern))
        {
            return ToolCategory::Edit;
        }
        if [
            "cat ", "grep ", "rg ", "ls ", "find ", "head ", "tail ", "diff ",
        ]
        .iter()
        .any(|pattern| command.contains(pattern))
        {
            return ToolCategory::Read;
        }
    }
    ToolCategory::Other
}

fn error_severity(text: &str) -> f64 {
    let text = text.to_ascii_lowercase();
    if [
        "out of memory",
        "memoryerror",
        "cannot allocate memory",
        "connection refused",
        "econnrefused",
    ]
    .iter()
    .any(|pattern| text.contains(pattern))
    {
        return 1.0;
    }
    if [
        "traceback (most recent call last)",
        "modulenotfounderror:",
        "importerror:",
        "assertionerror",
        "valueerror:",
        "syntaxerror:",
        "timed out",
        "deadline exceeded",
        "filenotfounderror:",
        "no such file or directory",
    ]
    .iter()
    .any(|pattern| text.contains(pattern))
    {
        return 0.7;
    }
    if ["exit code 1", "exit status 1", "returned non-zero"]
        .iter()
        .any(|pattern| text.contains(pattern))
    {
        return 0.3;
    }
    0.0
}

fn test_passed(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    let has_pass = [
        " passed",
        "passed in",
        "tests passed",
        "test result: ok",
        "\nok ",
    ]
    .iter()
    .any(|pattern| text.contains(pattern));
    let has_failure = ["assertionerror", "fatal:", "error:", "✗ "]
        .iter()
        .any(|pattern| text.contains(pattern));
    has_pass && !has_failure
}

#[cfg(test)]
mod tests {
    use super::{DecisionSource, PickerMode, Tier, select_tier};
    use dynamo_protocols::types::ChatCompletionRequestMessage;

    fn messages(json: &str) -> Vec<ChatCompletionRequestMessage> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn no_tool_history_uses_picker_default() {
        let messages = messages(r#"[{"role":"user","content":"fix it"}]"#);
        let decision = select_tier(&messages, PickerMode::EfficientFirst, 0.5, 3);
        assert_eq!(decision.tier, Tier::Efficient);
        assert_eq!(decision.source, DecisionSource::FallOpen);
    }

    #[test]
    fn critical_tool_error_forces_capable() {
        let messages = messages(
            r#"[
              {"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"exec_command","arguments":"{\"command\":\"pytest\"}"}}]},
              {"role":"tool","tool_call_id":"1","content":"out of memory"}
            ]"#,
        );
        let decision = select_tier(&messages, PickerMode::EfficientFirst, 0.99, 3);
        assert_eq!(decision.tier, Tier::Capable);
        assert_eq!(decision.source, DecisionSource::Override);
    }

    #[test]
    fn passing_tests_after_edit_selects_efficient() {
        let messages = messages(
            r#"[
              {"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"apply_patch","arguments":"{}"}}]},
              {"role":"tool","tool_call_id":"1","content":"Done"},
              {"role":"assistant","tool_calls":[{"id":"2","type":"function","function":{"name":"exec_command","arguments":"{\"command\":\"pytest\"}"}}]},
              {"role":"tool","tool_call_id":"2","content":"12 passed in 1.2s"}
            ]"#,
        );
        let decision = select_tier(&messages, PickerMode::CapableFirst, 0.99, 3);
        assert_eq!(decision.tier, Tier::Efficient);
        assert_eq!(decision.source, DecisionSource::TestsPassed);
    }
}
