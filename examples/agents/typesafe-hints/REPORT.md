<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## TypeSafe-to-Dynamo Agent Hints: Implementation and Validation

Date: September 17, 2026. Status: functional proof of concept; not a performance
benchmark or production qualification.

## Outcome

A TypeSafe-backed proxy generated serving hints, forwarded real chat requests to
Dynamo with SGLang on one H100 80 GB, and returned both non-streaming and streaming
responses. Explicit caller overrides survived forwarding. The frontend logged a
speculative next-turn prefill for one inferred-positive request.

This establishes an integration path, not that inferred hints improve latency,
throughput, fairness, or cache efficiency. H200, B200, and B300 were not tested.

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

Local packaging checks: 15 tests passed; Python lint/format and Bash syntax checks
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

Before making performance or production claims:

1. Compare no hints, static hints, and TypeSafe hints on the same trace, seed,
   model, concurrency, and GPU allocation.
2. Measure classification latency separately from generation TTFT and total
   latency; collect p50/p95/p99, tokens/second, failures, and TypeSafe cost.
3. Introduce contention and mixed priorities; check starvation and whether
   strict priority delivers the intended scheduling behavior.
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
