<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## TypeSafe-to-Dynamo Agent Hints: Implementation and Validation

Run dates: September 17–18, 2026 (UTC). Status: functional proof of concept with synthetic
32B and historical 0.6B experiments; not production qualification.

## Does it improve anything?

**The larger model shows a prioritization benefit, not a general speedup.**
In the exploratory cache-reset 192-request test, TypeSafe reduced urgent mean
TTFT by 24.0%, but increased overall latency by 14.4% and reduced output rate
by 6.9%. Static application hints had a lower urgent mean TTFT than TypeSafe.
Online hints made low-load completion latency 7.6% worse. The nominal urgent-latency gain
at 128 requests is order-sensitive and disappears when the worker cache is reset
before each arm: overall latency then worsens by 15.8%, with no observed queue.
The 256-request comparison failed on fallback. The controls below are more
informative than the original pooled average.

The historical, artificially capped 0.6B microbenchmark did show 34.8% better
urgent mean Time To First Token (TTFT), alongside worse overall performance.
That narrow result is preserved below; it is not the larger-model conclusion.

Recommendation: use application-provided hints when intent is already known.
Consider semantic inference only when priority cannot be supplied directly and
the value of prioritizing urgent work outweighs the extra delay for other work.
This implementation should not be presented as a general Dynamo speedup.

## Larger-Model Rerun

The rerun serves **Qwen3-32B-FP8 (32.8 billion parameters)**, pinned to revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`. This is about 55 times the parameter
count of the original 0.6B model. It is still the *serving model*, not a
competitor to TypeSafe: no hints, application hints, and TypeSafe hints all use
this same model, precision, worker, and GPU.

The successful deployment uses one scheduler-classified B300 PCIe allocation.
The device reports the generic name `NVIDIA Graphics Device`, 146,687 MiB of
memory, and driver 610.57.04. Do not treat this as a characterization of every
commercial B300 SKU. The runtime automatically chose a 392,720-token BF16 KV
cache and a 4,096-request scheduler ceiling. **There is no manually imposed
four-request limit or KV-memory override.** Both benchmark HTTP connection pools
are unlimited so they do not silently throttle the larger bursts.

Inputs contain approximately 2,040 tokens and every measured generation produces
256 tokens. Six repetitions counterbalance all three-arm orders at each load.
See the [protocol](BENCHMARK.md#larger-model-rerun-protocol),
[sweep script](deploy/run_benchmark_sweep.sh), and
[environment and verified source hashes](results/environment-32b.json).
The traces remain synthetic, with explicit urgency cues and warm-cache controls;
larger weights do not turn this into a production trace or answer-quality test.

### Low Load

All 144 measured requests completed successfully without fallback, with 48 per
variant and no observed queue.

| Metric | No hints | Static hints | Online TypeSafe hints |
| --- | ---: | ---: | ---: |
| Mean TTFT, ms | 51.4 | 51.5 | 279.1 |
| Mean total latency, ms | 3,013.7 | 3,014.4 | 3,242.2 |
| Achieved output tokens/second | 84.9 | 84.9 | 79.0 |

TypeSafe adds 228.5 ms to mean total latency, a **7.6% regression**, with a
paired-bootstrap interval of 7.2–7.9% slower. Evaluation averages 235.9 ms.
Its relative completion-time overhead is smaller than in the tiny-model test,
but first-token latency still increases substantially. A larger model alone
did not make online hints beneficial in this no-queue workload.

Evidence: [raw results](results/benchmark-32b-low-load.json),
[summary](results/summary-32b-low-load.json), and
[queue samples](results/telemetry-32b-low-load.json).

### Burst Sweep: Inspect Order Effects Before Claiming a Win

The completed load points each contain six repetitions of all three variants.
All measured generations succeeded without TypeSafe fallback at these levels.

| Burst size | Urgent TTFT: no hints, ms | Static, ms | TypeSafe, ms | TypeSafe overall latency change |
| --- | ---: | ---: | ---: | ---: |
| 8 | 76 | 58 | 290 | 8.2% slower |
| 32 | 89 | 87 | 730 | 27.1% slower |
| 128 | 3,928 | 918 | 2,057 | 10.2% slower |

The no-hints worker had no observed queue at 8 or 32 requests. At 128, sampled
queue peaks were 114, 114, and 112 for no hints, static hints, and TypeSafe,
respectively. Each reached 128 running requests; there was no four-request cap.

The 128-request TypeSafe mean suggests a 47.6% urgent-TTFT reduction, **but only
two of six paired repetitions improved**. Its paired-bootstrap interval spans
−2,639.7% to +83.5%, so this is not a reliable improvement estimate. No-hints
urgent TTFT was about 11.5–11.6 seconds in the two repetitions where it ran
first, versus approximately 0.10 seconds in the other four. The other variants
also exhibited a first-arm penalty. Counterbalancing did not make these
individual paired comparisons immune to persistent state. A separate
cache-reset control below addresses this carryover before drawing a conclusion.

Inspect the per-repetition records, not just the pooled averages:

- 8 requests: [raw](results/benchmark-32b-burst-8.json),
  [summary](results/summary-32b-burst-8.json), [telemetry](results/telemetry-32b-burst-8.json).
- 32 requests: [raw](results/benchmark-32b-burst-32.json),
  [summary](results/summary-32b-burst-32.json), [telemetry](results/telemetry-32b-burst-32.json).
- 128 requests: [raw](results/benchmark-32b-burst-128.json),
  [summary](results/summary-32b-burst-128.json), [telemetry](results/telemetry-32b-burst-128.json).

### Cache-Reset Control

The follow-up clears the exclusive worker's KV cache through its supported RPC
before **every** arm, verifies a success acknowledgment, then performs the same
warm-up. It does not alter worker capacity, TypeSafe thresholds, or request
budgets. The control was added after inspecting the sweep and is exploratory.
Initial payload hashes match the original 128-request sweep exactly.

At 128 requests, all 2,304 generations succeeded without fallback. Each variant
reached 128 running requests, with no observed queue. The apparent urgent-latency
benefit in the original sweep **did not survive this control**.

| Metric, reset before each arm | No hints | Static hints | Online TypeSafe hints |
| --- | ---: | ---: | ---: |
| Urgent mean TTFT, ms | 101.1 | 137.4 | 1,252.3 |
| Urgent p95 TTFT, ms | 123.2 | 334.5 | 1,433.6 |
| All-request mean latency, ms | 8,771.6 | 8,762.5 | 10,158.1 |
| Achieved output tokens/second | 3,603.0 | 3,598.9 | 3,047.3 |

TypeSafe was slower in all six paired repetitions. Overall latency increased
15.8% (paired-bootstrap interval: 15.1–16.5% slower), and achieved output rate
fell 15.4%. Evaluation averaged 1,094.0 ms during the burst. These results
support treating the original 47.6% pooled urgent-TTFT reduction as an
order-sensitive observation, not a validated speedup.

Evidence: [raw](results/benchmark-32b-reset-128.json),
[summary](results/summary-32b-reset-128.json), and
[telemetry](results/telemetry-32b-reset-128.json).

At **192 requests**, all 3,456 generations succeeded without fallback. Queues
were observed: sampled peaks were 32, 25, and 21 requests for no hints, static,
and TypeSafe, respectively; each variant reached 189 running requests.

| Metric, reset before each arm | No hints | Static hints | Online TypeSafe hints |
| --- | ---: | ---: | ---: |
| Urgent mean TTFT, ms | 3,730.0 | 2,628.7 | 2,836.3 |
| Urgent p95 TTFT, ms | 12,783.3 | 12,415.8 | 2,803.6 |
| All-request mean latency, ms | 12,499.1 | 12,261.1 | 14,303.7 |
| Achieved output tokens/second | 2,681.7 | 2,699.8 | 2,496.4 |

TypeSafe improved urgent mean TTFT in five of six paired repetitions, by
**24.0%** overall (paired-bootstrap 95% interval: **5.9–37.0%**). However,
all-request latency was **14.4% worse** (interval: 11.4–17.7% worse), with no
improved repetition, and achieved output rate fell 6.9%. Evaluation averaged
1,540.7 ms. Static hints had lower mean urgent TTFT and lower overall latency
than TypeSafe; its urgent improvement versus no hints was 29.5%, with an
interval spanning −0.4% to +58.7%. The p95 values are descriptive pooled
statistics, not independently validated tail-latency guarantees.

This supports a narrow latency tradeoff under this synthetic queued workload.
Because this load was selected after inspecting the main sweep, six repetitions
and an exploratory bootstrap interval are not production qualification or a
multiple-comparison-adjusted confirmation. It does not isolate the contribution
of each hint from inference-induced arrival timing.

Evidence: [raw](results/benchmark-32b-reset-192.json),
[summary](results/summary-32b-reset-192.json), and
[telemetry](results/telemetry-32b-reset-192.json).

Across the completed main suites and both reset controls, **9,216 measured
requests** form valid comparisons, with zero generation errors or fallbacks.
The additional 768 generations from the incomplete 256-request test below are
excluded from that count.

### Two-Turn Workload

All 288 measured requests succeeded without fallback. Each variant completed
48 two-turn conversations. All 96 corresponding message groups and output
groups matched across variants by hash. No worker queue was observed.

| Metric | No hints | Static hints | Online TypeSafe hints |
| --- | ---: | ---: | ---: |
| Mean latency per request, ms | 3,027.6 | 3,033.7 | 3,280.9 |
| Second-turn mean TTFT, ms | 43.1 | 42.5 | 294.8 |
| Speculative-prefill events | 0 | 48 | 96 |

TypeSafe increased mean request latency by **8.4%**, with a paired-bootstrap
interval of 8.1–8.6% slower. Evaluation averaged 251.1 ms. Speculation executed,
but no useful end-to-end TypeSafe benefit was established. As in the original
test, TypeSafe also speculated after the final turn, where no next request
arrived. Event counts are not cache-hit or saved-work measurements.

Evidence: [raw](results/benchmark-32b-multi-turn.json),
[summary](results/summary-32b-multi-turn.json), and
[telemetry](results/telemetry-32b-multi-turn.json).

### Failed 256-Request Comparison

The first three arms completed 768 generations, but **9 of 256 TypeSafe
requests used fallback (3.5%)**. The runner stopped, preserved the incomplete
artifact, and did not produce a valid comparison summary. Those requests must
not be counted as successful TypeSafe decisions or used to claim a speedup.

Only 247 evaluator responses were recorded. The original harness did not retain
exception types or HTTP statuses for failed evaluation calls; their precise
cause cannot be recovered from this artifact. Follow-up instrumentation records
these sanitized fields without changing the policy or deadline. The independent
two-turn suite was continued separately after this failure.

Evidence: [incomplete 256-request run](results/benchmark-32b-burst-256.json).

### Startup Failures, Kept Separate from Performance

Two H100 80 GB allocations with driver 595.58.03 failed before any measured arm
completed. Host diagnostics reported Xid 79, “GPU has fallen off the bus,” and
Xid 154, “Node Reboot Required.” The first passed short live streaming checks
before failing on the longer-input warm-up. These are not evidence for or
against the hint policy. No hardware reset or reboot was attempted; both
unusable allocations were released after preserving diagnostic evidence.
See the [sanitized failure record](results/startup-failure-32b.json).

The replacement deployment passed [live verification](results/verify-live-32b.json).
Startup downloads and compilation are excluded from measured request timing.

## Historical 0.6B Microbenchmark Results

The comparison used one H100 80 GB, the same Qwen3-0.6B model and frontend, and
SGLang with `--max-running-requests 4 --schedule-policy fcfs` plus priority
scheduling. The capacity limit deliberately creates a queue; it is not the
H100's maximum-throughput configuration. Each generation produced exactly 128
tokens. Every variant used the same proxy handler. Six repetitions covered all
six run-order permutations. All 864 measured requests succeeded, with no
TypeSafe fallbacks. The run finished at 21:21:29 UTC on September 17, 2026.

### Urgent Work Under Contention

Each burst contained 18 background requests followed by six urgent requests.
There were 144 requests per variant, including 36 urgent requests.

| Metric | No hints | Static application hints | Online TypeSafe hints |
| --- | ---: | ---: | ---: |
| Urgent mean TTFT, ms | 953 | 177 | 622 |
| Urgent p95 TTFT, ms | 1,213 | 317 | 809 |
| Urgent mean total latency, ms | 1,152 | 376 | 825 |
| Background mean TTFT, ms | 433 | 640 | 1,241 |
| All-request mean total latency, ms | 763 | 724 | 1,291 |
| Achieved output tokens/second | 2,212 | 2,277 | 1,536 |

The TypeSafe urgent-TTFT improvement occurred in all six paired repetitions.
A paired bootstrap over repetition means gives a descriptive 95% interval of
27.8–40.8% improvement. Urgent total latency improved by 28.4%. This is a
redistribution of service toward urgent requests, not a throughput improvement:
background TTFT became 2.86 times the baseline. Static hints reduced urgent TTFT
by 81.4%, without an evaluation call, but also delayed background requests.

The first no-hints burst was slower than later repetitions. Keeping all runs
gives the numbers above. Excluding the entire first paired repetition from all
three variants still gives a 32.0% TypeSafe urgent-TTFT reduction; the direction
does not depend on that first burst.

Numeric [worker telemetry](results/benchmark-telemetry.json) confirms real
contention: each variant reached 20 queued requests with four running requests.
These are log-sampled peaks, not time-weighted queue statistics. Router pending
queue tiers and multi-worker load balancing were not isolated or validated.

### Control: Evaluate, Then Discard the Hints

A follow-up control retained the full online TypeSafe evaluation but discarded
its hints before forwarding. Six alternating paired repetitions added 288
requests, all successful without fallback. The same four-request worker limit,
24-request burst, and fixed token budget were retained.

Urgent mean TTFT was 1,091 ms when hints were discarded versus 594 ms when
applied: a 45.6% reduction, with a descriptive paired-bootstrap interval of
40.6–49.5%, and an improvement in all six pairs. Both arms reached 20 queued
requests. This supports a contribution from applying the hints, not merely from
delaying requests while calling TypeSafe. It still does not isolate individual
hint fields or remove external-service timing variation between trials.

Inspect the [control's raw results](results/benchmark-ablation.json),
[summary](results/ablation-summary.json), and
[queue evidence](results/ablation-telemetry.json). This separate control does not
replace the main three-arm comparison or its overall regressions.

### Low Load and Two-Turn Work

| Metric | No hints | Static application hints | Online TypeSafe hints |
| --- | ---: | ---: | ---: |
| Low-load mean TTFT, ms | 40.1 | 40.0 | 256.7 |
| Low-load mean total latency, ms | 205.4 | 204.9 | 441.4 |
| Two-turn workload mean latency per request, ms | 218.7 | 218.7 | 484.8 |
| Second-turn mean TTFT, ms | 23.8 | 23.5 | 294.0 |

Low load had 48 requests per variant. The two-turn workload had 48 conversations
(96 requests) per variant with a 250 ms simulated tool delay. Neither workload
had an observed worker queue. TypeSafe evaluation averaged 234 ms at low load,
432 ms in bursts, and 264 ms in the two-turn workload, including HTTP transport.

Speculation did execute: the frontend logged 48 speculative-prefill actions for
static hints and 96 for TypeSafe. TypeSafe requested speculation after both turns,
including 48 final turns with no subsequent request in this benchmark. Static
hints used knowledge of the fixed session boundary. No useful end-to-end
speculation benefit was established in this warm-cache, small-model workload.

### What Was and Was Not Demonstrated

TypeSafe assigned priority 0 to background requests and 7 to urgent requests in
these explicit synthetic prompts; static hints used 0 and 10. This is not an
accuracy evaluation on ambiguous or held-out real agent traffic. Static hints
also know the forced output length and session boundary. They are a strong
application-metadata baseline, not another classifier.

The benchmark leaves the four-question policy and thresholds unchanged and
includes the current per-call evaluation HTTP-session creation. The static
control deliberately bypasses TypeSafe; passing explicit hints to the normal
prototype proxy still incurs evaluation. The benchmark therefore compares
architectural choices, not three public proxy configuration modes.

The KV cache is not flushed between arms; initial prompts are warmed equally,
and order is counterbalanced. All 96 corresponding two-turn message inputs
matched across variants by hash; 95 of 96 corresponding outputs matched. Future
runs must audit this again because follow-up messages contain generated output.
The first-burst effect, six-repetition sample, explicit class cues, small model,
fixed generation lengths, and artificially capped concurrency limit
generalization. No H200/B200/B300, large-model, cold-cache, or production SLO
claim follows from this result. OSL routing benefit cannot be established with
one worker; priority cache eviction was not tested under memory pressure.

Main-matrix TypeSafe usage was 467,295 input tokens and 36,576 output tokens across
288 evaluations, excluding warm-ups. No dollar-cost claim is made.

Reproduce and inspect:

- [Benchmark protocol and commands](BENCHMARK.md)
- [Raw requests, timings, judgments, and usage](results/benchmark-h100.json)
- [Aggregated metrics and paired intervals](results/benchmark-summary.json)
- [Benchmark runner](src/typesafe_agent_hints/benchmark.py) and
  [analyzer](src/typesafe_agent_hints/analyze_benchmark.py)

## Initial Functional Validation

A TypeSafe-backed proxy generated serving hints, forwarded real chat requests to
Dynamo with SGLang on one H100 80 GB, and returned both non-streaming and streaming
responses. Explicit caller overrides survived forwarding. The frontend logged a
speculative next-turn prefill for one inferred-positive request.

This initial smoke test established only the integration path. The controlled
comparison above was added afterward to measure benefits and regressions.

## Architecture and Policy

```text
Chat request
  -> TypeSafe System One (four questions in one call)
  -> deterministic validation, thresholds, token cap, caller overrides
  -> Dynamo frontend: nvext.agent_hints
  -> SGLang worker
  -> original JSON or server-sent event response
```

Implementation: [policy](src/typesafe_agent_hints/policy.py),
[TypeSafe client](src/typesafe_agent_hints/typesafe.py), and
[HTTP proxy](src/typesafe_agent_hints/server.py).

| Judgment | TypeSafe output | Deterministic mapping |
| --- | --- | --- |
| Serving priority | Choice | background 0, standard 3, interactive 7, urgent 10 |
| Strict queue priority | Noul probability | 1 at probability ≥0.80; otherwise 0 |
| Output sequence length (OSL) | Choice | tiny 64, short 256, long 1024, very_long 4096 |
| Speculative prefill | Noul probability | true at probability ≥0.70 |

OSL is capped by a positive integer `max_completion_tokens`, or `max_tokens`
when the former is absent. All answer types, choice labels, and probabilities
are validated. Probabilities must be finite and in [0, 1]; booleans are rejected
as probabilities. Choice confidence is preserved for inspection but does not yet
gate priority/OSL decisions. Thresholds and buckets are hand-selected, not calibrated.

The inference state includes messages, tools, tool choice, token limits, and stream
mode. It does not include trusted harness metadata, deadlines, tenant policy, or
queue state. Caller-provided hints override inference independently per field.
Typed outputs constrain shape; they do not establish semantic correctness.

Transport errors, timeouts, and invalid answers use conservative defaults unless
fail-open is disabled. Error details are reduced to the exception class in preview
output. Unexpected programming errors are not converted to fallback.

## Historical 0.6B Environment

| Component | Observed value |
| --- | --- |
| GPU | One NVIDIA H100 80 GB |
| Scheduling | Single-node Slurm GPU allocation on Computelab |
| Driver / CUDA container | 610.57.04 / 13.0.1 |
| Dynamo / backend | Published 1.4.2 SGLang runtime |
| Image digest | `sha256:3692c0c6a1ae23045f48384a09b539dcb9347ffa66ead5d0426aa90e85865012` |
| Model | `Qwen/Qwen3-0.6B` |
| Downloaded model revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| TypeSafe model | `jev-latest` (mutable alias) |
| Discovery | File |
| Frontend | KV routing, output-block tracking, KV events disabled |
| SGLang | Priority scheduling; priority radix eviction; page size 16; tensor parallelism 1 |

The live experiment ran the published runtime, not a build of this branch's Dynamo
core. Both worker and frontend needed the same writable Hugging Face cache
(`HF_HOME`); the frontend was restarted with that setting during bring-up.
The packaged launcher captures the corrected configuration. Python dependencies
and the TypeSafe model alias are not fully pinned, so repeated results may vary.

## Evidence

Raw response snapshots: [smoke-test.json](results/smoke-test.json) and
[verify-live.json](results/verify-live.json). These were captured during the live
experiment before branch packaging.

| Check | Observed result | What it establishes |
| --- | --- | --- |
| Interactive outage request | priority 10, strict 0, OSL 64, speculative true | Live inferred hints and a real completion |
| Background summary request | priority 3, strict 0, OSL 40, speculative false | Distinct inference and caller token-cap handling |
| Explicit overrides, non-streaming | priority 5, strict 1, OSL 32, speculative false; “Hello!” | Override preservation and JSON completion |
| Explicit overrides, streaming | Same hints; three data events, “Hello!”, terminal `[DONE]` | Server-sent event forwarding and completion |

The “background” request receiving standard priority is an observed judgment, not
a correctness claim. The short smoke-test generation limits produced truncated
reasoning text; those cases establish plumbing, not task-answer quality.

A sanitized frontend log excerpt from the live run:

```text
2026-09-17T19:15:53.602176Z
dynamo_llm::preprocessor::speculative_prefill
Speculative prefill: sending next-turn prefix num_tokens=132
```

This proves a speculative action was triggered. It does not prove the next turn
reused the prefix or benefited from it. Private hostnames, user paths, and
credentials are excluded from the branch; complete operational logs are not published.

The recorded end-to-end samples were 0.293 seconds (non-streaming) and 0.254 seconds
(streaming). These are two isolated observations with different request modes,
not Time To First Token (TTFT), percentiles, or an A/B comparison.

## Packaging Validation

The branch adds isolated policy and local HTTP tests, including malformed
TypeSafe answers, fallback, fail-closed mode, caller precedence, invalid input,
and server-sent event byte preservation. The local suite does not call TypeSafe
or load a GPU model. See [reproduction instructions](README.md).

Local packaging checks: 34 tests passed; Python lint/format and Bash syntax checks
passed. Scoped documentation lint reported zero errors; the earlier full scan
had five pre-existing navigation warnings elsewhere. The Fern CLI is not installed in the local
environment, so a full site build and Fern broken-link check were not run.

Packaging hardening after the live run narrows fallback exceptions, defaults the
proxy to loopback, validates the `nvext` envelope, filters connection-specific
headers, and ensures upstream cleanup when downstream preparation fails.
These changes are covered by local checks. The original smoke snapshots predate
packaging; the 32B rerun executes the packaged launcher and implementation.
Its source hashes were checked against the deployed files. Benchmark-specific
HTTP connection limits are explicit in the protocol.

## Remaining Validation

Before generalizing these measurements or making production claims:

1. Repeat the three-arm comparison on held-out real traces, additional model
   families, and sustained arrival-rate distributions.
2. Measure sustained load, service-level-objective goodput, evaluation cost,
   and enough independent trials for reliable tail estimates.
3. Check starvation over long mixed-priority runs and isolate router strict
   priority from backend priority and admission timing.
4. Measure OSL error against actual output tokens and test routing under load.
5. Replay multi-turn tool workflows; measure prefill reuse, wasted GPU work,
   cache occupancy, and latency with speculation enabled versus disabled.
6. Add trusted application metadata, priority authorization, privacy controls,
   evaluator connection reuse, load tests, and authenticated ingress.
7. Pin evaluation/model/dependency versions and validate additional GPU families
   only when matching allocations are available.

## References

- [Dynamo agent hints contract](https://docs.nvidia.com/dynamo/agents/agent-hints)
- [TypeSafe](https://typesafe.ai/)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)
