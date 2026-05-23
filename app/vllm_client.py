from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.config import settings
from app.models import ChatCompletionRequest, ChatCompletionResponse


class VLLMClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.vllm_base_url,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._client

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = await self._get_client()
        payload = request.model_dump(exclude_none=True, mode="json")

        # Remove extra_body and merge into payload
        if request.extra_body:
            for k, v in request.extra_body.items():
                payload[k] = v
            payload.pop("extra_body", None)

        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return ChatCompletionResponse(**data)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        client = await self._get_client()
        payload = request.model_dump(exclude_none=True, mode="json")
        payload["stream"] = True

        if request.extra_body:
            for k, v in request.extra_body.items():
                payload[k] = v
            payload.pop("extra_body", None)

        async with client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line

    async def models(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        resp = await client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


vllm_client = VLLMClient()