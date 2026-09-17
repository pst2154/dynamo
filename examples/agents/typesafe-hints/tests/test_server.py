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

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from typesafe_agent_hints.policy import HintDecision
from typesafe_agent_hints.server import create_app

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.timeout(10),
]


class FakeEngine:
    async def infer(self, request):
        return HintDecision(
            {
                "priority": 7,
                "strict_priority": 0,
                "osl": 64,
                "speculative_prefill": True,
            },
            {"test": True},
        )


@pytest.mark.asyncio
async def test_preview_is_inspectable_and_preserves_explicit_hints():
    app = create_app(FakeEngine(), "http://unused.invalid")
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/agent-hints/preview",
            json={"messages": [], "nvext": {"agent_hints": {"priority": 99}}},
        )
        body = await response.json()
    assert response.status == 200
    assert body["merged_hints"]["priority"] == 99
    assert body["merged_hints"]["osl"] == 64


@pytest.mark.asyncio
async def test_proxy_forwards_enriched_request_and_json_response():
    received = {}

    async def upstream_handler(request):
        received.update(await request.json())
        return web.json_response({"ok": True})

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream_handler)
    async with TestServer(upstream_app) as upstream:
        app = create_app(FakeEngine(), str(upstream.make_url("/")))
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            body = await response.json()
            header = json.loads(response.headers["x-typesafe-agent-hints"])
    assert response.status == 200
    assert body == {"ok": True}
    assert received["nvext"]["agent_hints"]["priority"] == 7
    assert header["speculative_prefill"] is True


@pytest.mark.parametrize(
    "payload", [[], {"nvext": "invalid"}, {"nvext": {"agent_hints": []}}]
)
async def test_invalid_payload_is_rejected(payload):
    app = create_app(FakeEngine(), "http://unused.invalid")
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status == 400


async def test_proxy_preserves_sse_bytes_and_done_marker():
    data = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    async def upstream_handler(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(data[:20])
        await response.write(data[20:])
        await response.write_eof()
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream_handler)
    async with TestServer(upstream_app) as upstream:
        app = create_app(FakeEngine(), str(upstream.make_url("/")))
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", json={"stream": True})
            assert await response.read() == data
            assert response.content_type == "text/event-stream"
