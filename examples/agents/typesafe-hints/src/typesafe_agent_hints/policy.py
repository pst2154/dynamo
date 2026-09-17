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

"""TypeSafe questions and deterministic policy for Dynamo agent hints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from aiohttp import ClientError

PRIORITY_VALUES = {
    "background": 0,
    "standard": 3,
    "interactive": 7,
    "urgent": 10,
}

OSL_VALUES = {
    "tiny": 64,
    "short": 256,
    "long": 1024,
    "very_long": 4096,
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "priority": {
        "type": "choice",
        "instructions": (
            "Choose the serving priority for `request`. Judge latency importance, not task "
            "difficulty or emotional wording. Prefer standard unless the application state "
            "clearly justifies another tier."
        ),
        "criteria": {
            "background": "Deferrable batch, speculative, maintenance, or offline work.",
            "standard": "Ordinary work without a concrete latency requirement.",
            "interactive": "A person or agent loop is actively blocked waiting for this response.",
            "urgent": "A time-critical operation where delay has immediate operational impact.",
        },
    },
    "strict_priority": {
        "type": "noul",
        "instructions": (
            "Must this request always precede ordinary pending requests because a person or "
            "time-critical agent workflow is immediately blocked?"
        ),
        "criteria": {
            "true": "Delay has a concrete immediate cost; strict queue precedence is justified.",
            "false": "Normal priority ordering is sufficient.",
        },
    },
    "output_length": {
        "type": "choice",
        "instructions": (
            "Estimate the likely number of generated output tokens for `request`, independent "
            "of its input length. Use explicit requested format and token limits when present."
        ),
        "criteria": {
            "tiny": "About 64 tokens: label, short answer, or one compact tool call.",
            "short": "About 256 tokens: a few paragraphs or a modest structured response.",
            "long": "About 1024 tokens: detailed analysis, code, or a substantial report.",
            "very_long": "About 4096 tokens: extensive code, document, or multi-section output.",
        },
    },
    "speculative_prefill": {
        "type": "noul",
        "instructions": (
            "Is this request part of a multi-turn agent or tool-calling loop where another turn "
            "is likely soon and will reuse this conversation prefix?"
        ),
        "criteria": {
            "true": "A predictable next turn is likely soon, so warming its prefix is useful.",
            "false": "This is likely one-shot, the next turn is uncertain, or reuse is unlikely.",
        },
    },
}


class Evaluator(Protocol):
    async def evaluate(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class HintDecision:
    """Inferred hints plus inspectable TypeSafe answers."""

    hints: dict[str, Any]
    answers: dict[str, Any]
    source: str = "typesafe"


def _request_token_limit(request: dict[str, Any]) -> int | None:
    raw = request.get("max_completion_tokens", request.get("max_tokens"))
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return None


def fallback_hints(request: dict[str, Any]) -> dict[str, Any]:
    """Conservative hints used when TypeSafe is unavailable."""

    return {
        "priority": 3,
        "strict_priority": 0,
        "osl": _request_token_limit(request) or 256,
        "speculative_prefill": False,
    }


def answers_to_hints(
    answers: dict[str, Any],
    request: dict[str, Any],
    *,
    strict_threshold: float = 0.80,
    prefill_threshold: float = 0.70,
) -> dict[str, Any]:
    """Translate probabilistic judgments into Dynamo's typed serving contract."""

    for name, question in QUESTIONS.items():
        answer = answers.get(name)
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError(f"Missing or invalid answer: {name}")
        field = "confidence" if question["type"] == "choice" else "noul"
        value = answer.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid probability: {name}")
        if (
            question["type"] == "choice"
            and answer.get("choice") not in question["criteria"]
        ):
            raise ValueError(f"Unknown choice: {name}")

    priority_answer = answers.get("priority", {})
    length_answer = answers.get("output_length", {})
    strict_answer = answers.get("strict_priority", {})
    prefill_answer = answers.get("speculative_prefill", {})

    priority = PRIORITY_VALUES.get(priority_answer.get("choice"), 3)
    osl = OSL_VALUES.get(length_answer.get("choice"), 256)
    limit = _request_token_limit(request)
    if limit is not None:
        osl = min(osl, limit)

    strict_probability = strict_answer.get("noul", 0.0)
    prefill_probability = prefill_answer.get("noul", 0.0)
    strict = (
        isinstance(strict_probability, (int, float))
        and strict_probability >= strict_threshold
    )
    prefill = (
        isinstance(prefill_probability, (int, float))
        and prefill_probability >= prefill_threshold
    )

    return {
        "priority": priority,
        "strict_priority": 1 if strict else 0,
        "osl": max(1, osl),
        "speculative_prefill": prefill,
    }


def existing_hints(request: dict[str, Any]) -> dict[str, Any]:
    nvext = request.get("nvext")
    if not isinstance(nvext, dict):
        return {}
    hints = nvext.get("agent_hints")
    return dict(hints) if isinstance(hints, dict) else {}


def merge_hints(request: dict[str, Any], inferred: dict[str, Any]) -> dict[str, Any]:
    """Merge hints without overriding explicit caller intent."""

    merged = dict(inferred)
    merged.update(existing_hints(request))
    return merged


def enrich_request(request: dict[str, Any], inferred: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(request)
    nvext = dict(enriched.get("nvext") or {})
    nvext["agent_hints"] = merge_hints(request, inferred)
    enriched["nvext"] = nvext
    return enriched


class HintEngine:
    """Batch four independent serving judgments into one TypeSafe request."""

    def __init__(self, evaluator: Evaluator, *, fail_open: bool = True) -> None:
        self._evaluator = evaluator
        self._fail_open = fail_open

    async def infer(self, request: dict[str, Any]) -> HintDecision:
        state = {
            "request": {
                "messages": request.get("messages", []),
                "tools": request.get("tools", []),
                "tool_choice": request.get("tool_choice"),
                "max_tokens": request.get("max_tokens"),
                "max_completion_tokens": request.get("max_completion_tokens"),
                "stream": request.get("stream", False),
            }
        }
        try:
            response = await self._evaluator.evaluate(state, QUESTIONS)
            answers = response.get("answers")
            if not isinstance(answers, dict):
                raise ValueError("TypeSafe response omitted answers")
            return HintDecision(answers_to_hints(answers, request), answers)
        except (ClientError, TimeoutError, ValueError) as error:
            if not self._fail_open:
                raise
            return HintDecision(
                fallback_hints(request),
                {"error": type(error).__name__},
                source="fallback",
            )
