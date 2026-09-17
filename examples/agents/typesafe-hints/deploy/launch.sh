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

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
: "${TYPESAFE_API_KEY:?Export TYPESAFE_API_KEY before launch}"

# Only terminate this launcher's children, not the caller's process group.
CHILD_PIDS=()
cleanup() {
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export DYN_HTTP_PORT="${DYN_HTTP_PORT:-8000}"
export DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
RUNTIME_DIR="$(mktemp -d)"
export HF_HOME="${HF_HOME:-$RUNTIME_DIR/huggingface}"
export DYN_FILE_KV="$RUNTIME_DIR/discovery"
mkdir -p "$DYN_FILE_KV"
# Keep the image's Dynamo/backend dependencies; isolate proxy installation.
python3 -m venv --system-site-packages "$RUNTIME_DIR/venv"
"$RUNTIME_DIR/venv/bin/python" -m pip install "$SCRIPT_DIR/.."
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
CAPACITY_ARGS=()
if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
    CAPACITY_ARGS+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
fi
print_launch_banner "TypeSafe agent hints with SGLang" "$MODEL" "$DYN_HTTP_PORT"

python3 -m dynamo.frontend --discovery-backend file \
    --router-mode kv --no-router-kv-events --router-track-output-blocks \
    --http-port "$DYN_HTTP_PORT" &
CHILD_PIDS+=("$!")

# GPU_MEM_ARGS intentionally expands into CLI flags from the shared helper.
# shellcheck disable=SC2086
python3 -m dynamo.sglang --model-path "$MODEL" --served-model-name "$MODEL" \
    --discovery-backend file --enable-priority-scheduling \
    --radix-eviction-policy priority --page-size 16 --tp 1 \
    --disable-piecewise-cuda-graph --schedule-policy fcfs \
    "${CAPACITY_ARGS[@]}" $GPU_MEM_ARGS &
CHILD_PIDS+=("$!")

"$RUNTIME_DIR/venv/bin/python" -m typesafe_agent_hints.server \
    --upstream "http://127.0.0.1:$DYN_HTTP_PORT" &
CHILD_PIDS+=("$!")
wait_any_exit
