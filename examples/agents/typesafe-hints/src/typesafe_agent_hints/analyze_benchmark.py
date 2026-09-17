# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Summarize complete paired benchmark runs without treating requests as independent."""

import argparse
import json
import random
import statistics
from pathlib import Path

from typesafe_agent_hints.benchmark import percentile, summarize


def paired_improvement(baseline, variant, seed=17):
    if len(baseline) != len(variant) or not baseline:
        raise ValueError("Expected equal, nonempty paired measurements")
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        base = statistics.mean(baseline[i] for i in indices)
        other = statistics.mean(variant[i] for i in indices)
        draws.append(100 * (base - other) / base)
    return {
        "pairs": len(baseline),
        "baseline_mean_ms": statistics.mean(baseline),
        "variant_mean_ms": statistics.mean(variant),
        "improvement_percent": 100
        * (1 - statistics.mean(variant) / statistics.mean(baseline)),
        "paired_bootstrap_95_percent": [
            percentile(draws, 0.025),
            percentile(draws, 0.975),
        ],
        "improved_repetitions": sum(v < b for b, v in zip(baseline, variant)),
    }


def analyze(data):
    if "finished_at" not in data:
        raise ValueError(
            "Incomplete benchmark; do not summarize it as a finished experiment"
        )
    runs = data["runs"]
    rounds = data["configuration"]["rounds"]
    modes = data["configuration"].get("modes", ["none", "static", "typesafe"])
    grouped = {}
    for run in runs:
        key = (run["suite"], run["mode"])
        grouped.setdefault(key, []).append(run)
        if run["summary"]["errors"] or run["summary"]["fallbacks"]:
            raise ValueError(
                "Errors/fallbacks invalidate the measured TypeSafe comparison"
            )
        if any(
            row["output_tokens"] != data["configuration"]["tokens"]
            for row in run["records"]
        ):
            raise ValueError("Unequal generated token counts")
    suites = sorted({key[0] for key in grouped})
    result = {"configuration": data["configuration"], "suites": {}, "comparisons": []}
    for suite in suites:
        result["suites"][suite] = {}
        for mode in modes:
            arm = sorted(grouped[(suite, mode)], key=lambda r: r["repeat"])
            if [r["repeat"] for r in arm] != list(range(rounds)):
                raise ValueError("Missing or duplicated repetitions")
            result["suites"][suite][mode] = summarize(
                [row for run in arm for row in run["records"]],
                sum(run["summary"]["wall_seconds"] for run in arm),
            )
            result["suites"][suite][mode]["evaluation_usage"] = {
                field: sum(
                    (e.get("usage") or {}).get(field, 0)
                    for run in arm
                    for e in run["evaluations"]
                )
                for field in ("input_tokens", "output_tokens")
            }
        for baseline_mode, variant_mode in (
            ("none", "static"),
            ("none", "typesafe"),
            ("static", "typesafe"),
            ("typesafe_no_hints", "typesafe"),
        ):
            if baseline_mode not in modes or variant_mode not in modes:
                continue
            baseline = sorted(
                grouped[(suite, baseline_mode)], key=lambda r: r["repeat"]
            )
            variant = sorted(grouped[(suite, variant_mode)], key=lambda r: r["repeat"])
            if any(
                b["payload_sha256"] != v["payload_sha256"]
                for b, v in zip(baseline, variant)
            ):
                raise ValueError("Arms did not use identical initial workload payloads")
            for group in ("all", "urgent", "background", "turn_2"):
                if group not in baseline[0]["summary"]:
                    continue
                for metric in ("ttft_ms", "latency_ms"):
                    paired = paired_improvement(
                        [r["summary"][group][metric]["mean"] for r in baseline],
                        [r["summary"][group][metric]["mean"] for r in variant],
                    )
                    result["comparisons"].append(
                        {
                            "suite": suite,
                            "baseline": baseline_mode,
                            "variant": variant_mode,
                            "group": group,
                            "metric": metric,
                            **paired,
                        }
                    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for suite, modes in result["suites"].items():
        for mode, summary in modes.items():
            print(
                suite,
                mode,
                "n=",
                summary["requests"],
                "mean_ttft_ms=",
                round(summary["all"]["ttft_ms"]["mean"], 1),
                "mean_latency_ms=",
                round(summary["all"]["latency_ms"]["mean"], 1),
                "urgent_ttft_ms=",
                round(summary["urgent"]["ttft_ms"]["mean"], 1),
                "output_tokens/s=",
                round(summary["output_tokens_per_second"], 1),
            )


if __name__ == "__main__":
    main()
