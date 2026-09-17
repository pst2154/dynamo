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
and the forced output length. It uses priority 10/0, strict priority 1/0, OSL 128,
and speculation on the first turn of the multi-turn suite. It is a strong
application-metadata baseline, not an inferred classifier.

TypeSafe uses the same implementation as the proxy, including creating a new
evaluation HTTP session per call. Evaluation is not precomputed or removed from
end-to-end timing. A warm-up evaluation is excluded, but every measured request
still incurs its own call. Errors or fallbacks invalidate the comparison; the
runner preserves partial output and stops.

## Workloads

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

The KV cache is not flushed between arms. Rotation balances order effects, but
the two-turn suite is a warm-cache experiment and cannot establish cold-cache
benefit. Follow-up input includes model output, which can vary despite a fixed
seed; per-request message/output hashes support auditing this difference.

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

`--worker-max-running-requests` records the verified deployment setting; it does
not change the server. Confirm the worker starts with
`--max-running-requests 4 --schedule-policy fcfs --enable-priority-scheduling`.
Do not run against a shared production endpoint.

Inspect worker queue telemetry to verify actual contention. Keep the same
deployment for every variant, retain full result JSON, and report regressions
alongside improvements. The analyzer rejects partial matrices, failed/fallback
arms, unequal output-token counts, and mismatched initial workloads.
