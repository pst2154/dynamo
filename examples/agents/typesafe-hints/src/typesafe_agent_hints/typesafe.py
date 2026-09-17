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

"""Minimal asynchronous client for TypeSafe System One."""

from __future__ import annotations

import ssl
from typing import Any

import certifi
from aiohttp import ClientSession, ClientTimeout, TCPConnector


class TypeSafeEvaluator:
    def __init__(
        self,
        api_key: str,
        *,
        api_url: str = "https://api.typesafe.ai/v1/systemone",
        model: str = "jev-latest",
        timeout_seconds: float = 15.0,
        session: ClientSession | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("TYPESAFE_API_KEY is required")
        self._api_key = api_key
        self._api_url = api_url
        self._model = model
        self._timeout = ClientTimeout(total=timeout_seconds)
        self._session = session

    async def evaluate(
        self, state: dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        owned_session = self._session is None
        connector = None
        if owned_session:
            connector = TCPConnector(
                ssl=ssl.create_default_context(cafile=certifi.where())
            )
        session = self._session or ClientSession(
            timeout=self._timeout, connector=connector
        )
        try:
            async with session.post(
                self._api_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"state": state, "model": self._model, "questions": questions},
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                payload = await response.json()
                if not isinstance(payload, dict):
                    raise ValueError("TypeSafe returned a non-object response")
                return payload
        finally:
            if owned_session:
                await session.close()
