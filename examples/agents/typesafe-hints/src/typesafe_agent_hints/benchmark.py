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


"""Counterbalanced serving benchmark, including online TypeSafe overhead.

Runs all modes through the same proxy handler against one exclusive deployment.
Use only synthetic/approved prompts: the TypeSafe arm sends them externally.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import os
import socket
import statistics
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import (
    ClientError,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    TCPConnector,
    web,
)
from typesafe_agent_hints.policy import HintDecision, HintEngine
from typesafe_agent_hints.server import create_app
from typesafe_agent_hints.typesafe import TypeSafeEvaluator


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(records, elapsed):
    good = [r for r in records if r["ok"]]
    result = {
        "requests": len(records),
        "successful": len(good),
        "errors": len(records) - len(good),
        "wall_seconds": elapsed,
        "output_tokens_per_second": sum(r["output_tokens"] for r in good) / elapsed,
        "fallbacks": sum(r.get("source") == "fallback" for r in records),
    }
    for group in ("all", "urgent", "background", "turn_2"):
        rows = [
            r
            for r in good
            if group == "all"
            or r["class"] == group
            or (group == "turn_2" and r["turn"] == 2)
        ]
        if not rows:
            continue
        result[group] = {"count": len(rows)}
        for metric in ("ttft_ms", "latency_ms", "hint_ms"):
            values = [r[metric] for r in rows]
            result[group][metric] = {
                "mean": statistics.mean(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
            }
    return result


class RecordingEvaluator:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    async def evaluate(self, state, questions):
        start = time.perf_counter()
        try:
            response = await self.delegate.evaluate(state, questions)
        except (ClientError, TimeoutError, ValueError) as error:
            self.calls.append(
                {
                    "state_sha256": fingerprint(state),
                    "elapsed_ms": (time.perf_counter() - start) * 1000,
                    "error_type": type(error).__name__,
                    "http_status": error.status
                    if isinstance(error, ClientResponseError)
                    else None,
                }
            )
            raise
        self.calls.append(
            {
                "state_sha256": fingerprint(state),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "model": response.get("model"),
                "usage": response.get("usage"),
                "answers": response.get("answers"),
            }
        )
        return response


class TimedEngine:
    def __init__(self, mode, evaluator):
        self.mode = mode
        self.engine = HintEngine(evaluator)
        self.records = {}

    async def infer(self, request):
        start = time.perf_counter()
        if self.mode.startswith("typesafe"):
            decision = await self.engine.infer(request)
        else:
            # Static hints are supplied by the harness; no model call in either control.
            decision = HintDecision({}, {}, self.mode)
        self.records[fingerprint(request["messages"])] = {
            "hint_ms": (time.perf_counter() - start) * 1000,
            "source": decision.source,
            "inferred_hints": decision.hints,
            "fallback_error": decision.answers.get("error")
            if decision.source == "fallback"
            else None,
        }
        if self.mode == "typesafe_no_hints":
            return HintDecision({}, decision.answers, decision.source)
        return decision


@asynccontextmanager
async def proxy_endpoint(engine, upstream):
    runner = web.AppRunner(
        create_app(engine, upstream, connection_limit=0), access_log=None
    )
    await runner.setup()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 0))
            sock.setblocking(False)
            port = sock.getsockname()[1]
            site = web.SockSite(runner, sock)
            await site.start()
            yield f"http://127.0.0.1:{port}"
        finally:
            await runner.cleanup()


def make_payload(
    model, repeat, index, tokens, urgent, multi_turn=False, padding_lines=None
):
    # The same serialized payload is reused across arms within a repetition.
    trace = hashlib.sha256(f"{repeat}:{index}".encode()).hexdigest()
    context = (
        "An incident operator is blocked waiting for this answer to restore a "
        "production checkout service that is currently unavailable. "
        if urgent
        else "This is a deferrable overnight archive maintenance batch. No person or "
        "online workflow is waiting; completion can wait until tomorrow. "
    )
    task = (
        "You are in a multi-turn debugging agent loop. After this answer, a tool "
        "result will arrive and you will immediately continue using this history. "
        if multi_turn
        else "This is a one-shot request; no follow-up turn is planned. "
    )
    padding = "Synthetic log entry: cache lookup succeeded and worker is healthy. "
    if padding_lines is None:
        padding_lines = 160 if multi_turn else 8
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Trace {trace}. {context}{task}\n"
                + padding * padding_lines
                + "\nWrite a numbered diagnostic checklist. Continue until the token limit.",
            }
        ],
        "temperature": 0,
        "seed": 17,
        "max_tokens": tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def with_static_hints(payload, urgent, multi_turn):
    return {
        **payload,
        "nvext": {
            "agent_hints": {
                "priority": 10 if urgent else 0,
                "strict_priority": 1 if urgent else 0,
                "osl": payload["max_tokens"],
                "speculative_prefill": multi_turn,
            }
        },
    }


async def generate(session, endpoint, payload, engine, request_class, index, turn=1):
    start = time.perf_counter()
    first = None
    parts = []
    usage = {}
    done = False
    row = {
        "index": index,
        "class": request_class,
        "turn": turn,
        "ok": False,
        "messages_sha256": fingerprint(payload["messages"]),
    }
    try:
        async with session.post(
            endpoint + "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            row["forwarded_hints"] = json.loads(
                response.headers.get("x-typesafe-agent-hints", "{}")
            )
            async for raw in response.content:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = (
                        delta.get("content") or delta.get("reasoning_content") or ""
                    )
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        parts.append(content)
        if not done or first is None or not usage.get("completion_tokens"):
            raise ValueError("Missing stream terminator, first token, or token usage")
        row.update(
            {
                "ok": True,
                "ttft_ms": (first - start) * 1000,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "output_tokens": usage["completion_tokens"],
                "input_tokens": usage.get("prompt_tokens"),
                "output_sha256": fingerprint("".join(parts)),
            }
        )
    except (ClientError, TimeoutError, ValueError, OSError) as error:
        row["error"] = type(error).__name__
    row.update(engine.records.get(fingerprint(payload["messages"]), {}))
    return row, "".join(parts)


def arrival_offset(index, count):
    """Keep urgent arrivals after the entire background burst at every load."""
    background_count = count * 3 // 4
    if index < background_count:
        return index * 0.005
    return max(0.15, background_count * 0.005) + (index - background_count) * 0.005


async def reset_cache(endpoint):
    """Run the optional Dynamo-native reset helper outside measured timing."""
    env = os.environ.copy()
    env.pop("DYN_SYSTEM_PORT", None)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "typesafe_agent_hints.reset_cache",
        "--endpoint",
        endpoint,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        async with asyncio.timeout(60):
            stdout, stderr = await process.communicate()
    except TimeoutError:
        process.kill()
        await process.communicate()
        raise
    if process.returncode != 0 or b"CACHE_RESET_OK" not in stdout.splitlines():
        raise RuntimeError(
            "Cache reset failed: " + stderr.decode(errors="replace")[-2000:]
        )


async def run_arm(args, suite, repeat, mode):
    started_at = datetime.now(timezone.utc).isoformat()
    if args.cache_reset_endpoint:
        await reset_cache(args.cache_reset_endpoint)
    evaluator = RecordingEvaluator(TypeSafeEvaluator(os.environ["TYPESAFE_API_KEY"]))
    engine = TimedEngine(mode, evaluator)
    count = args.requests if suite == "contention" else args.sessions
    payloads = [
        make_payload(
            args.model,
            repeat,
            i,
            args.tokens,
            i >= count * 3 // 4,
            suite == "multi_turn",
            args.padding_lines,
        )
        for i in range(count)
    ]
    timeout = ClientTimeout(total=args.request_timeout)
    async with ClientSession(
        timeout=timeout, connector=TCPConnector(limit=0)
    ) as session:
        # Warm every input equally in every arm; these requests are excluded.
        for payload in payloads:
            warm = {**payload, "max_tokens": 1, "stream": False}
            warm.pop("stream_options")
            async with session.post(
                args.upstream + "/v1/chat/completions", json=warm
            ) as response:
                response.raise_for_status()
                await response.read()
        if mode.startswith("typesafe"):
            await engine.infer(payloads[0])
            evaluator.calls.clear()
            engine.records.clear()
        async with proxy_endpoint(engine, args.upstream) as endpoint:
            start = time.perf_counter()

            async def one(index, payload):
                urgent = index >= count * 3 // 4
                if suite == "contention":
                    # Background arrives first; urgency has to overcome an existing queue.
                    offset = arrival_offset(index, count)
                    await asyncio.sleep(max(0, start + offset - time.perf_counter()))
                request = (
                    with_static_hints(payload, urgent, suite == "multi_turn")
                    if mode == "static"
                    else payload
                )
                row, content = await generate(
                    session,
                    endpoint,
                    request,
                    engine,
                    "urgent" if urgent else "background",
                    index,
                )
                rows = [row]
                if suite == "multi_turn" and row["ok"]:
                    await asyncio.sleep(args.think_seconds)
                    followup = {
                        **payload,
                        "messages": payload["messages"]
                        + [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": "Tool result: the worker is healthy. Continue the diagnostic checklist.",
                            },
                        ],
                    }
                    if mode == "static":
                        followup = with_static_hints(followup, urgent, False)
                    second, _ = await generate(
                        session, endpoint, followup, engine, rows[0]["class"], index, 2
                    )
                    rows.append(second)
                return rows

            if suite == "contention":
                batches = await asyncio.gather(
                    *(one(i, p) for i, p in enumerate(payloads))
                )
            else:
                batches = [await one(i, p) for i, p in enumerate(payloads)]
            elapsed = time.perf_counter() - start
    records = [r for batch in batches for r in batch]
    return {
        "suite": suite,
        "repeat": repeat,
        "mode": mode,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "payload_sha256": fingerprint(payloads),
        "cache_reset_before_warmup": bool(args.cache_reset_endpoint),
        "summary": summarize(records, elapsed),
        "records": records,
        "evaluations": evaluator.calls,
    }


async def main_async(args):
    output = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "model": args.model,
            "rounds": args.rounds,
            "requests": args.requests,
            "sessions": args.sessions,
            "tokens": args.tokens,
            "think_seconds": args.think_seconds,
            "worker_max_running_requests": args.worker_max_running_requests,
            "padding_lines": args.padding_lines,
            "request_timeout_seconds": args.request_timeout,
            "http_connection_limit": 0,
            "cache_reset_endpoint": args.cache_reset_endpoint,
            "arrival_schedule": "5ms spacing; urgent begins after background or 150ms, whichever is later",
            "modes": args.modes,
            "ordering": "all permutations cycled across repetitions",
            "warmup": "one unmeasured token per input before every arm",
        },
        "runs": [],
    }
    suites = (
        ["low_load", "contention", "multi_turn"]
        if args.suite == "all"
        else [args.suite]
    )
    orders = list(itertools.permutations(args.modes))
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for suite in suites:
        for repeat in range(args.rounds):
            for mode in orders[repeat % len(orders)]:
                run = await run_arm(args, suite, repeat, mode)
                output["runs"].append(run)
                args.output.write_text(json.dumps(output, indent=2) + "\n")
                print(
                    json.dumps(
                        {k: run[k] for k in ("suite", "repeat", "mode", "summary")}
                    ),
                    flush=True,
                )
                if run["summary"]["errors"] or run["summary"]["fallbacks"]:
                    raise RuntimeError("Benchmark arm failed; partial results retained")
    output["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "static", "typesafe"],
        choices=["none", "static", "typesafe", "typesafe_no_hints"],
    )
    parser.add_argument(
        "--suite",
        choices=["all", "low_load", "contention", "multi_turn"],
        default="all",
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--think-seconds", type=float, default=0.25)
    parser.add_argument(
        "--worker-max-running-requests",
        type=int,
        help="Configured worker override; omit when using automatic capacity",
    )
    parser.add_argument("--padding-lines", type=int)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--cache-reset-endpoint",
        help="Single exclusive Dynamo worker clear_kv_blocks endpoint; requires ai-dynamo and matching DYN_FILE_KV",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")
    if (
        min(
            args.rounds,
            args.requests,
            args.sessions,
            args.tokens,
        )
        <= 0
    ):
        parser.error("counts, token limits, and worker capacity must be positive")
    if (
        args.worker_max_running_requests is not None
        and args.worker_max_running_requests <= 0
    ):
        parser.error("worker capacity override must be positive")
    if args.padding_lines is not None and args.padding_lines < 0:
        parser.error("padding lines must be nonnegative")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("request timeout must be finite and positive")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
