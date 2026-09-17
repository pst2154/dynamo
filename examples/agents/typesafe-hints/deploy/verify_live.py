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

"""Verify streaming, caller overrides, and real TypeSafe-backed inference."""

import json
import os
import time
import urllib.request


def main():
    results = []
    for stream in (False, True):
        hints = {
            "priority": 5,
            "strict_priority": 1,
            "osl": 32,
            "speculative_prefill": False,
        }
        body = {
            "model": os.getenv("MODEL", "Qwen/Qwen3-0.6B"),
            "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": stream,
            "nvext": {"agent_hints": hints},
        }
        start = time.monotonic()
        url = (
            os.getenv("PROXY_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
            + "/v1/chat/completions"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            if json.loads(response.headers["x-typesafe-agent-hints"]) != hints:
                raise ValueError("Explicit hints changed")
            if response.headers["x-typesafe-hints-source"] != "typesafe":
                raise ValueError("Request used fallback instead of TypeSafe")
            if stream:
                chunks, done, parts = 0, False, []
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    chunks += 1
                    for choice in event.get("choices", []):
                        parts.append(choice.get("delta", {}).get("content") or "")
                content = "".join(parts)
                if not (done and chunks and content):
                    raise ValueError("Incomplete or empty stream")
            else:
                completion = json.load(response)
                content = completion["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError("Empty completion")
                chunks = 1
        results.append(
            {
                "stream": stream,
                "hints": hints,
                "source": "typesafe",
                "chunks": chunks,
                "content": content,
                "elapsed_seconds": round(time.monotonic() - start, 3),
            }
        )
    print(json.dumps({"status": "passed", "checks": results}, indent=2))


if __name__ == "__main__":
    main()
