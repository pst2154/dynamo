<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## TypeSafe Agent Hints

Experimental proxy that uses TypeSafe System One to infer NVIDIA Dynamo
`nvext.agent_hints` for chat completions. No Dynamo core changes are required.
The [validation report](REPORT.md) includes a Qwen3-32B-FP8 rerun on a
scheduler-classified B300 PCIe GPU, without an artificial concurrency cap.
At 192 requests with cache reset between arms, online hints reduced urgent mean
TTFT by 24.0%, but increased overall latency by 14.4% and reduced output rate
by 6.9%. Static hints had lower mean urgent TTFT; low-load TypeSafe was slower.
These are exploratory synthetic results, not a general speedup. See the
[benchmark protocol](BENCHMARK.md) and report for controls and failed runs.

## Run Against an Existing Dynamo Endpoint

Requires Python 3.11+, a TypeSafe API key, and a Dynamo endpoint supporting
[agent hints](https://docs.nvidia.com/dynamo/agents/agent-hints).
Run these commands from this example directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
read -r -s -p "TypeSafe API key: " TYPESAFE_API_KEY
export TYPESAFE_API_KEY
python -m typesafe_agent_hints.server --upstream http://127.0.0.1:8000
```

The proxy listens on loopback port 8001 by default. In another terminal:

```bash
curl --fail-with-body http://127.0.0.1:8001/v1/agent-hints/preview \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Give one debugging step for an outage"}],"max_tokens":64}'
python deploy/smoke_test.py
python deploy/verify_live.py
```

The live scripts default to `Qwen/Qwen3-0.6B`; override `MODEL` and
`PROXY_BASE_URL` for another deployment. They require real TypeSafe answers:
a fallback response fails verification. Successful output contains
`"status": "ok"` or `"status": "passed"`.

## Launch Dynamo on a GPU

Use Linux, Bash 4.3+, Docker with NVIDIA GPU support, an exclusively assigned GPU,
and internet access for the image, Python dependencies, model, and TypeSafe API.
The historical small-model experiment used an H100 80 GB; the 32B rerun used
a scheduler-classified B300 PCIe allocation. See the report for exact device
identity and separate H100 startup failures during the larger-model attempt.

From the repository root, with `TYPESAFE_API_KEY` exported:

```bash
docker run --rm --gpus all --shm-size=10g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8001:8001 \
  -e TYPESAFE_API_KEY -e HOST=0.0.0.0 \
  -v "$PWD:/workspace/dynamo:ro" \
  nvcr.io/nvidia/ai-dynamo/sglang-runtime@sha256:3692c0c6a1ae23045f48384a09b539dcb9347ffa66ead5d0426aa90e85865012 \
  bash /workspace/dynamo/examples/agents/typesafe-hints/deploy/launch.sh
```

The launcher installs the proxy in an ephemeral environment, starts the frontend,
SGLang worker, and proxy, and exits if any child exits. Wait until the frontend's
`/v1/models` lists the model (check inside the container) before running the live scripts. A proxy health check
alone does not prove model readiness.

On a Slurm cluster such as Computelab, run this only inside a GPU allocation.
Select a suitable available GPU partition and account using your site's scheduler
instructions; do not run the image pull or model server on a login node.
Use the same Docker command in the allocated compute shell with the checkout
available on that node. Do not use `--gpus all` on a shared node unless the site
runtime restricts the container to your allocated devices. The example does not
reserve GPUs or encode private partition/account names.

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | Required | TypeSafe credential; never commit it |
| `TYPESAFE_API_URL` | `https://api.typesafe.ai/v1/systemone` | Evaluation endpoint |
| `TYPESAFE_MODEL` | `jev-latest` | Evaluation model alias |
| `TYPESAFE_TIMEOUT_SECONDS` | `15` | Per-evaluation deadline |
| `TYPESAFE_FAIL_OPEN` | `true` | Use fallback on transport, timeout, or malformed-answer errors |
| `DYNAMO_BASE_URL` / `--upstream` | `http://127.0.0.1:8000` | Upstream frontend |
| `HOST` / `--host` | `127.0.0.1` | Bind address |
| `PORT` / `--port` | `8001` | Proxy port |
| `LOG_LEVEL` | `INFO` | Python logging level |

The launcher also accepts `MODEL`, `MODEL_PATH`, `LOG_DIR`, `DYN_HTTP_PORT`, `DYN_SYSTEM_PORT`,
`HF_HOME`, `DYN_FILE_KV`, optional `MAX_RUNNING_REQUESTS`, and the shared SGLang GPU-memory overrides. Defaults are the tested
model, port 8000, port 8081, and an ephemeral model cache.
`MODEL_PATH` can point to a pinned local snapshot while `MODEL` remains the served
API name. `LOG_DIR` optionally saves separate frontend, worker, and proxy logs;
otherwise logs go to standard output. See the
[larger-model protocol](BENCHMARK.md#larger-model-rerun-protocol) for the 32B run
without a worker concurrency override.
Use a fresh `DYN_FILE_KV` directory shared by the worker and benchmark client when
running the optional cache-reset control; never point it at a shared deployment.

Only `POST /v1/chat/completions` is forwarded. Streaming response bytes are
relayed without interpreting model output. Other endpoints are
`GET /healthz` (process liveness) and `POST /v1/agent-hints/preview` (inference
without generation). The preview exposes raw probabilities and the merged hints.
Each preview/completion makes a separate evaluation; their judgments can differ.

The completion response includes `x-typesafe-agent-hints` and
`x-typesafe-hints-source` (`typesafe` or `fallback`).
Explicit caller hints win field by field; Dynamo remains responsible for
validating their values. Other `nvext` fields are preserved.

## Security and Limitations

Use only with trusted callers and synthetic or approved data. Messages, tools,
tool choice, token limits, and stream mode are sent to the external TypeSafe
service. There is no authentication, tenant isolation, rate limiting, or priority
authorization. Prompt content can influence scheduling judgments. Do not expose
this example as a public endpoint or treat inferred urgency as an entitlement.

Fallback uses priority 3, strict priority 0, the caller's token limit or 256 for
output sequence length, and no speculative prefill. Programming errors propagate.
Disabling fail-open propagates evaluation failures as server errors.
A TypeSafe call still runs when all hints are explicitly supplied.

## Local Tests

No GPU or API key is required:

```bash
python -m pytest -q
```

Tests use fake evaluators and local ephemeral HTTP servers. Live evidence in
`results/` is a historical snapshot, not a result regenerated by this command.
