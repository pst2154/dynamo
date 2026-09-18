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


"""Regression tests for benchmark accounting and matched request generation."""

import pytest
from aiohttp import ClientResponseError, ClientSession
from typesafe_agent_hints.analyze_benchmark import analyze, paired_improvement
from typesafe_agent_hints.benchmark import (
    RecordingEvaluator,
    TimedEngine,
    arrival_offset,
    fingerprint,
    make_payload,
    percentile,
    proxy_endpoint,
    reset_cache,
    summarize,
    with_static_hints,
)
from typesafe_agent_hints.policy import HintDecision

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def test_percentiles_use_linear_interpolation():
    assert percentile([], 0.5) is None
    assert percentile([40, 10, 30, 20], 0.5) == 25
    assert percentile([10, 20], 0.95) == 19.5


@pytest.mark.parametrize("count", [8, 24, 32, 128, 256])
def test_urgent_arrivals_follow_background_at_every_load(count):
    background_count = count * 3 // 4
    assert arrival_offset(background_count, count) > arrival_offset(
        background_count - 1, count
    )
    assert arrival_offset(background_count, count) >= 0.15


def test_configurable_prompt_size_preserves_matched_inputs():
    short = make_payload("model", 1, 0, 256, False, padding_lines=8)
    long = make_payload("model", 1, 0, 256, False, padding_lines=160)
    assert len(long["messages"][0]["content"]) > len(short["messages"][0]["content"])
    assert long == make_payload("model", 1, 0, 256, False, padding_lines=160)


async def test_evaluator_records_sanitized_http_failure():
    class FailingEvaluator:
        async def evaluate(self, state, questions):
            raise ClientResponseError(None, (), status=429, message="not retained")

    evaluator = RecordingEvaluator(FailingEvaluator())
    with pytest.raises(ClientResponseError):
        await evaluator.evaluate({"synthetic": True}, {})
    record = evaluator.calls[0]
    assert record["error_type"] == "ClientResponseError"
    assert record["http_status"] == 429
    assert "not retained" not in str(record)


@pytest.mark.parametrize(
    "output,returncode,success",
    [
        (b"log\nCACHE_RESET_OK\n", 0, True),
        (b"CACHE_RESET_OK\n", 1, False),
        (b"", 0, False),
    ],
)
async def test_reset_requires_successful_process_and_acknowledgment(
    monkeypatch, output, returncode, success
):
    class Process:
        async def communicate(self):
            return output, b"diagnostic"

    process = Process()
    process.returncode = returncode

    async def spawn(*args, **kwargs):
        assert args[-1] == "dynamo.backend.clear_kv_blocks"
        assert "DYN_SYSTEM_PORT" not in kwargs["env"]
        return process

    monkeypatch.setenv("DYN_SYSTEM_PORT", "12345")
    monkeypatch.setattr(
        "typesafe_agent_hints.benchmark.asyncio.create_subprocess_exec", spawn
    )
    if success:
        await reset_cache("dynamo.backend.clear_kv_blocks")
    else:
        with pytest.raises(RuntimeError, match="Cache reset failed"):
            await reset_cache("dynamo.backend.clear_kv_blocks")


def test_static_control_preserves_model_work_and_original_payload():
    payload = make_payload("model", 1, 3, 128, True)
    before = fingerprint(payload)
    static = with_static_hints(payload, True, False)
    assert fingerprint(payload) == before
    assert {key: value for key, value in static.items() if key != "nvext"} == payload
    assert static["nvext"]["agent_hints"]["priority"] == 10
    assert static["nvext"]["agent_hints"]["osl"] == 128
    assert payload["ignore_eos"] is True


def test_summary_counts_errors_and_never_hides_fallback():
    good = {
        "ok": True,
        "class": "urgent",
        "turn": 2,
        "ttft_ms": 100,
        "latency_ms": 300,
        "hint_ms": 40,
        "output_tokens": 128,
        "source": "fallback",
    }
    result = summarize([good, {"ok": False}], 2)
    assert result["requests"] == 2
    assert result["errors"] == 1
    assert result["fallbacks"] == 1
    assert result["output_tokens_per_second"] == 64
    assert result["urgent"]["ttft_ms"]["p50"] == 100
    assert result["turn_2"]["count"] == 1


@pytest.mark.timeout(10)
async def test_benchmark_proxy_closes_its_listening_socket_cleanly():
    engine = TimedEngine("none", None)
    async with proxy_endpoint(engine, "http://unused.invalid") as endpoint:
        async with ClientSession() as session:
            async with session.get(endpoint + "/healthz") as response:
                assert response.status == 200


def test_analysis_rejects_partial_benchmarks():
    with pytest.raises(ValueError, match="Incomplete"):
        analyze({"runs": []})


def test_paired_improvement_reports_direction_and_interval():
    result = paired_improvement([100, 100, 100], [80, 80, 80])
    assert result["improvement_percent"] == pytest.approx(20)
    assert result["paired_bootstrap_95_percent"] == pytest.approx([20, 20])
    assert result["improved_repetitions"] == 3


async def test_ablation_discards_hints_but_retains_measured_inference():
    class StubEngine:
        async def infer(self, request):
            return HintDecision({"priority": 7}, {"test": True})

    engine = TimedEngine("typesafe_no_hints", None)
    engine.engine = StubEngine()
    request = {"messages": [{"role": "user", "content": "test"}]}
    decision = await engine.infer(request)
    assert decision.hints == {}
    assert decision.source == "typesafe"
    assert engine.records[fingerprint(request["messages"])]["inferred_hints"] == {
        "priority": 7
    }
