<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Benchmark Protocol

This benchmark asks whether online TypeSafe inference improves serving compared
with no hints and with hints supplied by an application that already knows its
workload. See [measured results](REPORT.md) before adopting the example.

## Controls

All variants use the same proxy HTTP handler, model, GPU, worker configuration,
streaming mode, temperature 0, seed 17, and fixed output-token budget.
The control engine performs no external evaluation.

| Variant | Hint source | External evaluation |
| --- | --- | --- |
| `none` | Empty hint object; backend defaults | None |
| `static` | Known synthetic workload class and token budget | None |
| `typesafe` | Existing four-question policy, unchanged thresholds | One online call per generation |

A separate two-arm control compares `typesafe_no_hints` (evaluate all questions,
record judgments, then discard the hints) with `typesafe`. It uses six repetitions
alternating the two possible orders. It tests whether applying hints helps beyond
the backend-arrival timing change caused by evaluation itself.

The static application knows which requests are urgent, which will continue,
and the forced output length. It uses priority 10/0, strict priority 1/0, the exact
output-token limit (128 originally, 256 in the larger-model rerun) for OSL,
and speculation on the first turn of the multi-turn suite. It is a strong
application-metadata baseline, not an inferred classifier.

TypeSafe uses the same implementation as the proxy, including creating a new
evaluation HTTP session per call. Evaluation is not precomputed or removed from
end-to-end timing. A warm-up evaluation is excluded, but every measured request
still incurs its own call. Errors or fallbacks invalidate the comparison; the
runner preserves partial output and stops.

## Larger-Model Rerun Protocol

The larger-model experiment uses `Qwen/Qwen3-32B-FP8`, pinned to revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`, on one scheduler-classified B300 PCIe
GPU (146,687 MiB reported memory). Two H100 startup attempts encountered GPU-loss
errors; see the [failure evidence](results/startup-failure-32b.json). Its 32.8 billion
parameters make it a serving workload rather than the original plumbing-scale
model. FP8 weight quantization is fixed across all arms; this experiment does
not measure its accuracy relative to BF16.

Inside the GPU container, download the pinned snapshot and launch from the
example directory with `TYPESAFE_API_KEY` securely supplied:

```bash
export MODEL=Qwen/Qwen3-32B-FP8
export MODEL_PATH
MODEL_PATH=$(python3 -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3-32B-FP8", revision="aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"))')
export LOG_DIR=/tmp/typesafe-32b-logs
bash deploy/launch.sh
```

After the frontend lists the model and the live verification passes, a separate
shell with the example installed can run the entire matrix:

```bash
MODEL=Qwen/Qwen3-32B-FP8 RESULT_DIR=/tmp/typesafe-32b-results \
  LOG_DIR=/tmp/typesafe-32b-logs bash deploy/run_benchmark_sweep.sh
```

The sweep refuses to overwrite existing raw results, stops on a failed arm,
and writes per-load analysis and queue evidence after each completed comparison.

Do not set `MAX_RUNNING_REQUESTS` or a GPU-memory override. Record the automatic
worker capacity from startup logs. Run six counterbalanced repetitions of each
three-arm comparison with 256 generated tokens and 160 synthetic padding lines
(approximately 2,000 input tokens). Sweep contention bursts of 8, 32, 128,
and 256 requests. The fourth level was specified before measurements because
the replacement GPU has more memory than the H100. Run low-load and two-turn
suites with eight sessions separately.
These are synthetic load points, not a production traffic distribution.

At every burst size, background arrivals are spaced 5 ms apart. Urgent requests
start after the background arrivals, or at 150 ms, whichever is later, also
spaced 5 ms apart. Both benchmark HTTP connection pools are unlimited, preventing
the default 100-connection pool from silently capping the larger load point.
The proxy's normal non-benchmark connection limit remains 100.

```bash
for count in 8 32 128 256; do
  python -m typesafe_agent_hints.benchmark \
    --model Qwen/Qwen3-32B-FP8 --suite contention --rounds 6 \
    --requests "$count" --tokens 256 --padding-lines 160 \
    --output "/tmp/benchmark-32b-burst-$count.json"
done
for suite in low_load multi_turn; do
  python -m typesafe_agent_hints.benchmark \
    --model Qwen/Qwen3-32B-FP8 --suite "$suite" --rounds 6 \
    --sessions 8 --tokens 256 --padding-lines 160 \
    --output "/tmp/benchmark-32b-$suite.json"
done
```

Analyze each file separately with `typesafe_agent_hints.analyze_benchmark` and
collect queue telemetry. A load point without a queue cannot demonstrate
priority scheduling under contention; report that result without manufacturing
a queue by restoring the four-request cap. Timing includes online inference.
Do not compare absolute throughput across the old and new model as a treatment
effect: model size, precision, input length, and output budget have changed.

## Original Microbenchmark Workloads

- **Low load:** Eight sequential requests per arm, six background and two urgent.
  No queue is intentionally created. This measures the overhead floor.
- **Contention:** A burst of 24 requests, 18 background followed by six urgent.
  Background requests start 5 ms apart; urgent requests start at 150 ms, then
  5 ms apart. Worker capacity is deliberately limited to four concurrent requests,
  with FCFS plus priority scheduling. This is a queueing stress test, not the
  H100's maximum-throughput configuration.
- **Two turns:** Eight sequential two-turn conversations per arm, about 2,000
  input tokens on turn one, with a fixed 250 ms simulated tool delay. The second
  request contains the first model response and a synthetic tool-result message.
  The TypeSafe arm evaluates both turns. This is not a real external tool execution.

All measured generations use `max_tokens=128` and `ignore_eos=true` to equalize
GPU output work. They do not test answer quality or natural stopping behavior.

Six repetitions cycle through all six permutations of the three variants. Every
initial input is warmed with one unmeasured output token before each arm.
The exact same initial payloads are used across arms within a repetition; hashes
are saved and checked by the analyzer. Requests are synthetically generated and
their class cues are explicit. This is not a held-out real agent trace.

The KV cache is not flushed between arms. Rotation counterbalances immediate
order, but persistent cache contents and eviction priorities can carry over
between arms and load points. In particular, warming each input does not
guarantee all warmed prefixes remain resident under memory pressure. These
warm-cache experiments cannot establish cold-cache benefit or isolate cache
eviction from queue priority. Follow-up input includes model output, which can vary despite a fixed
seed; per-request message/output hashes support auditing this difference.

### Follow-Up Cache-Reset Control

The 32B sweep exposed a large first-arm effect at 128 requests, and the
256-request run stopped on TypeSafe fallback. Consequently, a separate diagnostic
control was specified at 128 and 192 requests, with six counterbalanced
repetitions. This is an exploratory follow-up, not a pre-specified extension of
the original sweep. The intermediate load is below the failing 256-request
burst; worker capacity and the hint policy remain unchanged.

Before every arm, `--cache-reset-endpoint dynamo.backend.clear_kv_blocks` invokes
Dynamo's supported worker RPC in a separate client process and requires a
positive acknowledgment. It refuses discovery sets with more than one worker.
The reset precedes the same per-input warm-up and is excluded from timing.
This controls carryover between arms; it does not guarantee every warmed prefix
fits in memory at higher loads or measure a fully cold-cache workload.

Run the client inside the Dynamo image with `DYN_FILE_KV` set to the exact fresh
directory used by the exclusive worker. The optional reset helper requires
`ai-dynamo`; ordinary proxy use and CPU tests do not.

```bash
for count in 128 192; do
  python -m typesafe_agent_hints.benchmark \
    --model Qwen/Qwen3-32B-FP8 --suite contention --rounds 6 \
    --requests "$count" --tokens 256 --padding-lines 160 \
    --cache-reset-endpoint dynamo.backend.clear_kv_blocks \
    --output "/tmp/benchmark-32b-reset-$count.json"
done
```

Follow-up instrumentation also records sanitized evaluator exception types and
HTTP status codes. It does not retry, change policy, increase deadlines, or hide
fallbacks. The original failed 256-request artifact lacks those diagnostic fields
and must not be assigned a specific error cause retrospectively.

## Measurements

Time to first token (TTFT) starts immediately before the client sends the HTTP
request and ends on the first nonempty content/reasoning delta. Total latency
ends after the terminal `[DONE]` event. Both include proxy and TypeSafe overhead.
The runner separately records hint evaluation time, token usage, inferred hints,
and raw probabilities. It never writes the API key.

Achieved output tokens/second divides generated tokens by arm wall time,
including prescribed arrivals and tool delays. It is workload completion rate,
not a sustained maximum-throughput measurement. Warm-up work is excluded.

The analyzer pools request metrics for descriptive mean/p50/p95 summaries.
Comparisons use the six paired repetition means as the sampling units, with a
seeded 10,000-resample paired bootstrap interval. Requests in one burst are not
treated as independent trials. Six repetitions support only a narrow result;
intervals do not establish generalization to other models, loads, or GPUs.

## Reproduce

Use the [GPU launch instructions](README.md#launch-dynamo-on-a-gpu), adding
`-e MAX_RUNNING_REQUESTS=4` to the Docker invocation. Run the benchmark from an
environment that can reach the Dynamo frontend directly and has the example
package installed. For the supplied container launch, execute inside the
container using its installed virtual environment, or install the example into
a separate client environment on the allocated compute node.

Export `TYPESAFE_API_KEY` securely, then run:

```bash
python -m typesafe_agent_hints.benchmark \
  --upstream http://127.0.0.1:8000 \
  --suite all --rounds 6 --requests 24 --sessions 8 --tokens 128 \
  --worker-max-running-requests 4 \
  --output /tmp/benchmark-h100.json
python -m typesafe_agent_hints.analyze_benchmark /tmp/benchmark-h100.json \
  --output /tmp/benchmark-summary.json
```

Run the evaluate-and-discard control on the same deployment:

```bash
python -m typesafe_agent_hints.benchmark \
  --suite contention --rounds 6 --requests 24 --tokens 128 \
  --worker-max-running-requests 4 --modes typesafe_no_hints typesafe \
  --output /tmp/benchmark-ablation.json
python -m typesafe_agent_hints.analyze_benchmark /tmp/benchmark-ablation.json \
  --output /tmp/ablation-summary.json
```

`--worker-max-running-requests` records a deployment override; it does not change
the server. Omit it for automatic capacity (recorded as JSON `null`). For the
original microbenchmark only, confirm the worker starts with
`--max-running-requests 4 --schedule-policy fcfs --enable-priority-scheduling`.
Do not run against a shared production endpoint.

Inspect worker queue telemetry to verify actual contention. Keep the same
deployment for every variant, retain full result JSON, and report regressions
alongside improvements. The analyzer rejects partial matrices, failed/fallback
arms, unequal output-token counts, and mismatched initial workloads.
