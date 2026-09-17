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


"""Extract numeric queue/speculation evidence without publishing operational logs."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


def read_events(path):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    timestamp = re.compile(r"\d{4}-\d{2}-\d{2}T[0-9:.]+Z")
    events = []
    with path.open() as lines:
        for raw in lines:
            line = ansi.sub("", raw)
            stamp = timestamp.search(line)
            if stamp is None:
                continue
            queue = re.search(r"#queue-req: (\d+)", line)
            running = re.search(r"#running-req: (\d+)", line)
            speculative = "Speculative prefill: sending next-turn prefix" in line
            if queue is None and not speculative:
                continue
            events.append(
                {
                    "timestamp": datetime.fromisoformat(
                        stamp[0].replace("Z", "+00:00")
                    ),
                    "queue": int(queue[1]) if queue else None,
                    "running": int(running[1]) if running else None,
                    "speculative": speculative,
                }
            )
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--frontend-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.benchmark.read_text())
    events = read_events(args.worker_log) + read_events(args.frontend_log)
    summaries = []
    for run in data["runs"]:
        start = datetime.fromisoformat(run["started_at"])
        end = datetime.fromisoformat(run["finished_at"])
        samples = [e for e in events if start <= e["timestamp"] <= end]
        queues = [e["queue"] for e in samples if e["queue"] is not None]
        running = [e["running"] for e in samples if e["running"] is not None]
        summaries.append(
            {
                "suite": run["suite"],
                "repeat": run["repeat"],
                "mode": run["mode"],
                "queue_samples": len(queues),
                "positive_queue_samples": sum(n > 0 for n in queues),
                "max_queue_requests": max(queues, default=0),
                "max_running_requests": max(running, default=0),
                "speculative_prefill_events": sum(e["speculative"] for e in samples),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "description": "Timestamp-window log samples per arm, including warm-up. Counts are evidence of activity, not time-weighted queue metrics or cache-hit measurements.",
                "runs": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
