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

"""OpenAI-compatible proxy that enriches requests with Dynamo agent hints."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from .policy import HintEngine, enrich_request, existing_hints, merge_hints
from .typesafe import TypeSafeEvaluator

LOG = logging.getLogger("typesafe_agent_hints")
ENGINE_KEY = web.AppKey("engine", Any)
UPSTREAM_KEY = web.AppKey("upstream", str)
SESSION_KEY = web.AppKey("session", ClientSession)
CONNECTION_LIMIT_KEY = web.AppKey("connection_limit", int)
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def _forward_headers(headers: Any) -> dict[str, str]:
    connection_tokens = {
        token.strip().lower() for token in headers.get("Connection", "").split(",")
    }
    excluded = HOP_BY_HOP | connection_tokens
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


def _validate_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    nvext = payload.get("nvext")
    if nvext is not None and not isinstance(nvext, dict):
        raise web.HTTPBadRequest(text="nvext must be a JSON object")
    if (
        isinstance(nvext, dict)
        and "agent_hints" in nvext
        and not isinstance(nvext["agent_hints"], dict)
    ):
        raise web.HTTPBadRequest(text="agent_hints must be a JSON object")


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def preview(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="request body must be JSON")
    _validate_payload(payload)
    decision = await request.app[ENGINE_KEY].infer(payload)
    return web.json_response(
        {
            "explicit_hints": existing_hints(payload),
            "inferred_hints": decision.hints,
            "merged_hints": merge_hints(payload, decision.hints),
            "source": decision.source,
            "answers": decision.answers,
        }
    )


async def proxy(request: web.Request) -> web.StreamResponse:
    if request.method != "POST" or request.path != "/v1/chat/completions":
        raise web.HTTPNotFound()
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="request body must be JSON")
    _validate_payload(payload)

    decision = await request.app[ENGINE_KEY].infer(payload)
    enriched = enrich_request(payload, decision.hints)
    merged = enriched["nvext"]["agent_hints"]
    upstream_url = request.app[UPSTREAM_KEY] + request.path_qs
    session = request.app[SESSION_KEY]
    upstream = await session.post(
        upstream_url,
        headers={
            k: v
            for k, v in _forward_headers(request.headers).items()
            if k.lower() != "content-encoding"
        },
        json=enriched,
    )
    headers = _forward_headers(upstream.headers)
    headers["x-typesafe-agent-hints"] = json.dumps(merged, separators=(",", ":"))
    headers["x-typesafe-hints-source"] = decision.source
    downstream = web.StreamResponse(status=upstream.status, headers=headers)
    try:
        await downstream.prepare(request)
        async for chunk in upstream.content.iter_any():
            await downstream.write(chunk)
    finally:
        upstream.release()
    await downstream.write_eof()
    return downstream


async def _create_session(app: web.Application) -> None:
    app[SESSION_KEY] = ClientSession(
        timeout=ClientTimeout(total=None, connect=30),
        auto_decompress=False,
        connector=TCPConnector(limit=app[CONNECTION_LIMIT_KEY]),
    )


async def _close_session(app: web.Application) -> None:
    await app[SESSION_KEY].close()


def create_app(
    engine: HintEngine, upstream: str, *, connection_limit: int = 100
) -> web.Application:
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app[ENGINE_KEY] = engine
    app[UPSTREAM_KEY] = upstream.rstrip("/")
    app[CONNECTION_LIMIT_KEY] = connection_limit
    app.on_startup.append(_create_session)
    app.on_cleanup.append(_close_session)
    app.router.add_get("/healthz", health)
    app.router.add_post("/v1/agent-hints/preview", preview)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() not in {"0", "false", "no"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8001")))
    parser.add_argument(
        "--upstream", default=os.getenv("DYNAMO_BASE_URL", "http://127.0.0.1:8000")
    )
    args = parser.parse_args()

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    evaluator = TypeSafeEvaluator(
        os.environ["TYPESAFE_API_KEY"],
        api_url=os.getenv("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone"),
        model=os.getenv("TYPESAFE_MODEL", "jev-latest"),
        timeout_seconds=float(os.getenv("TYPESAFE_TIMEOUT_SECONDS", "15")),
    )
    engine = HintEngine(evaluator, fail_open=_env_bool("TYPESAFE_FAIL_OPEN", True))
    web.run_app(create_app(engine, args.upstream), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
