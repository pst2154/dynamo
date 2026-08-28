---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Frontend Model Router
---

# Frontend Model Router

The experimental frontend model router lets clients request a virtual model name and have the
Dynamo frontend select a concrete model. It is built into the frontend process; it does not require
a separate proxy, service, or routing pipeline.

Model routing and worker routing happen at different levels:

1. The model router maps a virtual model, such as `agent`, to a concrete model, such as
   `Qwen/Qwen3-8B`.
2. Dynamo's existing worker router selects a replica of that concrete model using the configured
   `--router-mode`.

Concrete model names that are not configured as routes continue to pass through unchanged.

## Configure the router

Create a TOML file that defines concrete targets and client-facing routes. The repository includes
[`components/src/dynamo/frontend/model-router.example.toml`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/frontend/model-router.example.toml)
as a starting point.

```toml
schema_version = 1

[targets.efficient]
id = "Qwen/Qwen3-0.6B"

[targets.capable]
id = "Qwen/Qwen3-8B"

[routes.fast]
type = "passthrough"
target = "efficient"

[routes.balanced]
type = "random"
targets = ["efficient", "capable"]
weights = [7.0, 3.0]

[routes.agent]
type = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = 0.5
recent_turn_window = 3
```

Target `id` values must match model display names registered with the frontend. A route uses its
TOML table name by default. Set an explicit `id` inside a route only when the client-facing name
should differ from the table name.

## Start the frontend

Pass the configuration file to the frontend:

```bash
python -m dynamo.frontend \
  --http-port 8000 \
  --model-router-config /path/to/model-router.toml
```

Alternatively, set the corresponding environment variable:

```bash
export DYN_FRONTEND_MODEL_ROUTER_CONFIG=/path/to/model-router.toml
python -m dynamo.frontend --http-port 8000
```

The frontend validates and compiles the configuration at startup. It does not start if the file is
unreadable, the TOML is invalid, a route references an unknown target, weights are invalid, or a
stage-router setting is outside its allowed range.

> [!NOTE]
> The configuration is loaded once at startup. Restart the frontend after changing it.

## Send requests

Use a route name in the standard request `model` field. For example, this request invokes the
`balanced` policy before normal Dynamo replica routing:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "balanced",
    "messages": [
      {"role": "user", "content": "Explain prefix caching."}
    ]
  }'
```

Configured route names also appear alongside concrete models in the models API:

```bash
curl http://localhost:8000/v1/models
```

## Policy reference

### Passthrough

A `passthrough` route always selects one concrete target. It is useful for stable aliases that let
operators change a backend model without changing clients.

```toml
[routes.fast]
type = "passthrough"
target = "efficient"
```

### Weighted random

A `random` route selects one target independently for each request. Omit `weights` to give all
targets equal probability. When present, every weight must be finite and greater than zero, and
there must be one weight per target.

```toml
[routes.balanced]
type = "random"
targets = ["efficient", "capable"]
weights = [7.0, 3.0]
```

Weights are relative: `[7.0, 3.0]` sends approximately 70 percent of requests to `efficient` and
30 percent to `capable` over a sufficiently large sample.

### Coding-agent stage router

A `stage_router` route chooses between efficient and capable model tiers. For Chat Completions, it
examines the conversation's tool-call history for coding-agent progress signals, including:

- Recent reads, edits, writes, and planning operations
- Tool errors and critical failures such as out-of-memory conditions
- Successful tests after implementation work
- Long exploratory or stalled sessions
- Compacted or continued sessions

Critical failures and compaction select the capable tier. Successful tests following recent edits
or writes select the efficient tier when no error is present. Other corroborating signals produce a
confidence score; the router changes tier only when it meets `confidence_threshold`.

```toml
[routes.agent]
type = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = 0.5
recent_turn_window = 3
```

The stage-router fields are:

| Field | Required | Description |
|---|---|---|
| `capable_target` | Yes | Target used for difficult, failed, or ambiguous work. |
| `efficient_target` | Yes | Target used for settled or lower-complexity work. |
| `picker` | Yes | Default tier: `capable_first` or `efficient_first`. |
| `confidence_threshold` | Yes | Minimum confidence from `0.0` through `1.0` for a signal-based decision. |
| `recent_turn_window` | No | Number of recent tool results to inspect. Defaults to `3`; must be at least `1`. |

The stage router uses the picker default when it has insufficient tool history or confidence.

## Protocol behavior

| API | Passthrough and random | Semantic stage routing |
|---|---|---|
| OpenAI Chat Completions | Supported | Supported |
| OpenAI Completions | Supported | Uses the configured picker default |
| OpenAI Responses | Supported | Uses the configured picker default |
| Anthropic Messages | Supported | Uses the configured picker default |

Semantic stage routing currently inspects OpenAI Chat Completions message and tool-call history.
Other supported APIs can use a stage route, but select its configured default tier.

## Observability

Each virtual-model selection produces a structured frontend log containing the route and selected
concrete model. Prometheus also exposes:

```text
dynamo_frontend_model_route_selections_total{route="balanced",target_model="Qwen/Qwen3-0.6B"}
```

If `--metrics-prefix` is configured, it replaces the default `dynamo` prefix.

## Current limitations

- Configuration reload requires a frontend restart.
- Targets are not checked for worker availability while parsing the configuration. Normal Dynamo
  readiness and routing behavior applies after a target is selected.
- Weighted-random selection is stateless and does not provide session affinity.
- Semantic stage signals are available only for OpenAI Chat Completions.
- The model router does not retry a request against another target after backend failure.

## See also

- [Frontend overview](README.md)
- [Frontend configuration reference](configuration.md)
- [Worker router](../router/README.md)
