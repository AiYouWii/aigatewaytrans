from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import settings
from app.models import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger("aigateway")

# Parameters that vLLM /v1/chat/completions does not support
_UNSUPPORTED_PARAMS = {"reasoning_effort", "extra_body"}


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

    def _build_payload(self, request: ChatCompletionRequest) -> dict:
        payload = request.model_dump(exclude_none=True, mode="json")

        # Merge extra_body into top-level, then remove the key
        if request.extra_body:
            for k, v in request.extra_body.items():
                payload[k] = v

        # Strip parameters vLLM doesn't recognize
        for key in _UNSUPPORTED_PARAMS:
            payload.pop(key, None)

        logger.debug("vLLM request payload: %s", payload)
        return payload

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = await self._get_client()
        payload = self._build_payload(request)

        resp = await client.post("/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            logger.error("vLLM error %d: %s", resp.status_code, resp.text)
        resp.raise_for_status()
        data = resp.json()
        return ChatCompletionResponse(**data)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        client = await self._get_client()
        payload = self._build_payload(request)
        payload["stream"] = True

        # Use a background task + asyncio.Queue so the httpx
        # async-with context is managed by the reader task, not
        # by the consumer generator.  This prevents GeneratorExit
        # from tearing down the httpx connection prematurely and
        # also ensures real-time streaming (no buffering delay).
        queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()

        async def _reader():
            try:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=payload
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        logger.error(
                            "vLLM stream error %d: %s",
                            response.status_code,
                            body.decode(),
                        )
                        response.raise_for_status()
                    async for line in response.aiter_lines():
                        await queue.put(line)
            except Exception as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)  # sentinel: stream finished

        task = asyncio.create_task(_reader())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

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