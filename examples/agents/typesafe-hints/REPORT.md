<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## TypeSafe-to-Dynamo Agent Hints: Implementation and Validation

Date: September 17, 2026. Status: functional proof of concept with a controlled
synthetic H100 benchmark; not production qualification.

## Does it improve anything?

**Yes, urgent-request latency under the tested queueing condition; no, overall
performance.** Online TypeSafe hints reduced urgent mean Time To First Token
(TTFT) from 953 ms to 622 ms, a **34.8% reduction including evaluation overhead**.
But average latency across all requests increased by 69.2%, and achieved output
tokens/second fell by 30.5%. Static application hints performed better than
TypeSafe on every measured workload.

Recommendation: use application-provided hints when intent is already known.
Consider semantic inference only when priority cannot be supplied directly and
the value of prioritizing urgent work outweighs the extra delay for other work.
This implementation should not be presented as a general Dynamo speedup.

## Controlled Benchmark Results

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

## Tested Environment

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

Local packaging checks: 22 tests passed; Python lint/format and Bash syntax checks
passed. Documentation lint reported no errors and five pre-existing navigation
warnings elsewhere in the repository. The Fern CLI is not installed in the local
environment, so a full site build and Fern broken-link check were not run.

Packaging hardening after the live run narrows fallback exceptions, defaults the
proxy to loopback, validates the `nvext` envelope, filters connection-specific
headers, and ensures upstream cleanup when downstream preparation fails.
These changes are covered by local checks; the saved GPU evidence is not a rerun
of this exact packaged revision. The new launcher is syntax-checked, not yet
re-executed as a complete container deployment.

## Remaining Validation

Before generalizing these measurements or making production claims:

1. Repeat the three-arm comparison on held-out real traces, larger models,
   default-capacity workers, and several arrival rates.
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
