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

"""Exercise live TypeSafe inference and a real Dynamo generation."""

from __future__ import annotations

import json
import os
import urllib.request


def post(path: str, body: dict) -> tuple[dict, dict[str, str]]:
    request = urllib.request.Request(
        os.getenv("PROXY_BASE_URL", "http://127.0.0.1:8001").rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response), {k.lower(): v for k, v in response.headers.items()}


def main() -> None:
    cases = [
        {
            "name": "interactive_agent",
            "request": {
                "model": os.getenv("MODEL", "Qwen/Qwen3-0.6B"),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an interactive coding agent. Use tools and continue until the bug is fixed.",
                    },
                    {
                        "role": "user",
                        "content": "The checkout API is down and customers are blocked. Give the next debugging step.",
                    },
                ],
                "max_tokens": 80,
                "stream": False,
            },
        },
        {
            "name": "background_summary",
            "request": {
                "model": os.getenv("MODEL", "Qwen/Qwen3-0.6B"),
                "messages": [
                    {
                        "role": "user",
                        "content": "When convenient, summarize these archived maintenance notes in one sentence.",
                    }
                ],
                "max_tokens": 40,
                "stream": False,
            },
        },
    ]
    results = []
    for case in cases:
        preview, _ = post("/v1/agent-hints/preview", case["request"])
        if preview["source"] != "typesafe":
            raise ValueError("Preview used fallback instead of TypeSafe")
        completion, headers = post("/v1/chat/completions", case["request"])
        if headers["x-typesafe-hints-source"] != "typesafe":
            raise ValueError("Completion used fallback instead of TypeSafe")
        if not completion.get("id") or not completion.get("choices"):
            raise ValueError("Missing completion ID or choices")
        message = completion["choices"][0]["message"]
        if not (
            message.get("content")
            or message.get("reasoning_content")
            or message.get("tool_calls")
        ):
            raise ValueError("Empty model response")
        hints = json.loads(headers["x-typesafe-agent-hints"])
        if not 0 < hints["osl"] <= case["request"]["max_tokens"]:
            raise ValueError("OSL outside request token cap")
        results.append(
            {
                "name": case["name"],
                "preview": preview,
                "forwarded_hints": hints,
                "completion_id": completion.get("id"),
                "completion_text": completion.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")[:240],
            }
        )
    print(json.dumps({"status": "ok", "cases": results}, indent=2))


if __name__ == "__main__":
    main()
