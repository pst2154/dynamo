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

import pytest
from typesafe_agent_hints.policy import HintEngine, answers_to_hints, enrich_request

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def sample_answers():
    return {
        "priority": {"type": "choice", "choice": "interactive", "confidence": 0.9},
        "strict_priority": {"type": "noul", "noul": 0.85},
        "output_length": {"type": "choice", "choice": "long", "confidence": 0.8},
        "speculative_prefill": {"type": "noul", "noul": 0.76},
    }


class FakeEvaluator:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def evaluate(self, state, questions):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return self.response


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {**sample_answers(), "strict_priority": {"type": "noul", "noul": float("nan")}},
        {
            **sample_answers(),
            "priority": {"type": "choice", "choice": "unknown", "confidence": 1},
        },
    ],
)
async def test_invalid_service_answers_are_marked_as_fallback(answers):
    decision = await HintEngine(FakeEvaluator({"answers": answers})).infer(
        {"messages": []}
    )
    assert decision.source == "fallback"
    assert decision.hints["speculative_prefill"] is False


def test_answers_map_to_dynamo_contract_and_respect_token_limit():
    hints = answers_to_hints(sample_answers(), {"max_tokens": 700})
    assert hints == {
        "priority": 7,
        "strict_priority": 1,
        "osl": 700,
        "speculative_prefill": True,
    }


def test_explicit_hints_win_field_by_field():
    request = {
        "messages": [],
        "nvext": {
            "agent_hints": {"priority": 42, "speculative_prefill": False},
            "agent_context": {"session_id": "keep-me"},
        },
    }
    enriched = enrich_request(request, answers_to_hints(sample_answers(), request))
    assert enriched["nvext"]["agent_hints"]["priority"] == 42
    assert enriched["nvext"]["agent_hints"]["speculative_prefill"] is False
    assert enriched["nvext"]["agent_hints"]["osl"] == 1024
    assert enriched["nvext"]["agent_context"] == {"session_id": "keep-me"}


@pytest.mark.asyncio
async def test_engine_batches_all_questions_in_one_call():
    evaluator = FakeEvaluator({"answers": sample_answers()})
    engine = HintEngine(evaluator)
    decision = await engine.infer({"messages": [{"role": "user", "content": "Help"}]})
    assert decision.source == "typesafe"
    assert decision.hints["priority"] == 7
    assert len(evaluator.calls) == 1
    assert set(evaluator.calls[0][1]) == {
        "priority",
        "strict_priority",
        "output_length",
        "speculative_prefill",
    }


@pytest.mark.asyncio
async def test_engine_fails_open_without_leaking_error_details():
    engine = HintEngine(FakeEvaluator(error=TimeoutError("secret-bearing detail")))
    decision = await engine.infer({"messages": [], "max_completion_tokens": 99})
    assert decision.source == "fallback"
    assert decision.hints["osl"] == 99
    assert decision.answers == {"error": "TimeoutError"}


async def test_programming_errors_are_not_hidden_by_fallback():
    engine = HintEngine(FakeEvaluator(error=RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        await engine.infer({})


async def test_fail_closed_propagates_timeout():
    engine = HintEngine(FakeEvaluator(error=TimeoutError()), fail_open=False)
    with pytest.raises(TimeoutError):
        await engine.infer({})
