---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Frontend
---

The Dynamo Frontend is the API gateway for serving LLM inference requests. It provides OpenAI-compatible HTTP endpoints and KServe gRPC endpoints, handling request preprocessing, routing, and response formatting.

## Feature Matrix

| Feature | Status |
|---------|--------|
| OpenAI Chat Completions API (`/v1/chat/completions`) | ✅ Supported |
| OpenAI Completions API (`/v1/completions`) | ✅ Supported |
| OpenAI Embeddings API (`/v1/embeddings`) | ✅ Supported |
| OpenAI Responses API (`/v1/responses`) | ✅ Supported |
| OpenAI Models API (`/v1/models`) | ✅ Supported |
| Image Generation (`/v1/images/generations`) | ✅ Supported |
| Video Generation (`/v1/videos/generations`) | ✅ Supported |
| Anthropic Messages API (`/v1/messages`) | 🧪 Experimental |
| KServe gRPC v2 API | ✅ Supported |
| Streaming responses (SSE) | ✅ Supported |
| Multi-model serving | ✅ Supported |
| Virtual model passthrough, weighted-random, and agent stage routes | 🧪 Experimental |
| Integrated KV-aware routing | ✅ Supported |
| Tool calling | ✅ Supported |
| TLS (HTTPS) | ✅ Supported |
| Swagger UI (`/docs`) | ✅ Supported |
| NVIDIA request extensions (`nvext`) | ✅ Supported |

## Quick Start

### Prerequisites

- Dynamo platform installed
- `etcd` and `nats-server -js` running
- At least one backend worker registered

### HTTP Frontend

```bash
python -m dynamo.frontend --http-port 8000
```

This starts an OpenAI-compatible HTTP server with integrated pre/post processing and routing. Backends are auto-discovered when they call `register_model`.

### Virtual model routes

The frontend includes a model router that exposes policy-backed virtual model names in addition to
concrete models discovered from workers. It runs in the same process and project as Dynamo's API
frontend and worker router. Start it with a native Dynamo routing configuration:

```bash
python -m dynamo.frontend \
  --http-port 8000 \
  --model-router-config components/src/dynamo/frontend/model-router.example.toml
```

Clients can then send a configured route ID such as `fast` or `smart` in the request's `model`
field. The frontend resolves the route to a concrete registered model before normal preprocessing
and replica routing. Passthrough, weighted-random, and signal-driven `stage_router` policies are
currently supported; concrete model names continue to work unchanged. The stage router examines
Chat Completions tool-call history, escalating critical failures and confident error/exploration
signals to the capable target while sending settled implementation work to the efficient target.

Selections are logged with the virtual route and concrete target and counted by
`dynamo_frontend_model_route_selections_total{route,target_model}` (or the configured metrics
prefix). See the [Frontend Model Router](model-router.md) guide for the complete configuration,
policy reference, request examples, protocol behavior, and current limitations.

The frontend does the pre and post processing. To do this it will need access to the model configuration files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, etc. It does not need the weights.

Frontend will download the files it needs from Hugging Face, no setup is required. However we recommend setting up [modelexpress-server](https://github.com/ai-dynamo/modelexpress) and a shared folder such as a Kubernetes PVC. This ensures the model is only downloaded once across the whole cluster.

If the model is not available on Hugging Face, such as a private or customized model, you will need to make the model files available locally at the same file path as on the backend. The backend's `--model-path <here>` will need to exist on the frontend and contain at least the configuration (JSON) files.

### KServe gRPC Frontend

```bash
python -m dynamo.frontend --kserve-grpc-server
```

See the [Frontend Guide](frontend-guide.md) for KServe-specific configuration and message formats.

### Kubernetes

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: frontend-example
spec:
  graphs:
    - name: frontend
      replicas: 1
      services:
        - name: Frontend
          image: nvcr.io/nvidia/dynamo/dynamo-vllm:latest
          command:
            - python
            - -m
            - dynamo.frontend
            - --http-port
            - "8000"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--http-port` | 8000 | HTTP server port |
| `--kserve-grpc-server` | false | Enable KServe gRPC server |
| `--router-mode` | `round-robin` | Routing strategy: `round-robin`, `random`, `kv`, `direct`, `least-loaded`, `device-aware-weighted` (`power-of-two` and `least-loaded` use synchronous prefill fallback in disaggregated prefill mode) |
| `--model-router-config` | unset | Dynamo TOML defining virtual model targets and passthrough, weighted-random, or stage-router policies. |

See the [Frontend Guide](frontend-guide.md) for full configuration options.

## Next Steps

| Document | Description |
|----------|-------------|
| [Configuration Reference](configuration.md) | All CLI arguments, env vars, and HTTP endpoints |
| [Frontend Guide](frontend-guide.md) | KServe gRPC configuration and integration |
| [Frontend Model Router](model-router.md) | Virtual models and model-selection policies |
| [NVIDIA Request Extensions (nvext)](nvext.md) | Custom request fields for routing hints and cache control |
| [Router Documentation](../router/README.md) | KV-aware routing configuration |
