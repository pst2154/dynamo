#!/bin/bash
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

# Run against an exclusive, ready deployment with automatic worker capacity.
set -euo pipefail
: "${MODEL:?Set MODEL to the deployed model name}"
: "${RESULT_DIR:?Set RESULT_DIR to a new output directory}"
: "${TYPESAFE_API_KEY:?Export TYPESAFE_API_KEY}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
mkdir -p "$RESULT_DIR"

run_case() {
    local name="$1"
    shift
    local output="$RESULT_DIR/benchmark-32b-$name.json"
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite $output" >&2
        exit 1
    fi
    "$PYTHON" -m typesafe_agent_hints.benchmark \
        --upstream "${DYNAMO_BASE_URL:-http://127.0.0.1:8000}" \
        --model "$MODEL" --rounds 6 --sessions 8 --tokens 256 \
        --padding-lines 160 --output "$output" "$@"
    "$PYTHON" -m typesafe_agent_hints.analyze_benchmark "$output" \
        --output "$RESULT_DIR/summary-32b-$name.json"
    if [[ -n "${LOG_DIR:-}" ]]; then
        "$PYTHON" "$SCRIPT_DIR/collect_telemetry.py" \
            --benchmark "$output" --worker-log "$LOG_DIR/worker.log" \
            --frontend-log "$LOG_DIR/frontend.log" \
            --output "$RESULT_DIR/telemetry-32b-$name.json"
    fi
}

run_case low-load --suite low_load
for count in 8 32 128 256; do
    run_case "burst-$count" --suite contention --requests "$count"
done
run_case multi-turn --suite multi_turn
