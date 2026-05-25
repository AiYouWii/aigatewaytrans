from __future__ import annotations

import json
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.conversation_store import store
from app.models import (
    ChatCompletionStreamChunk,
    ResponsesAPIRequest,
    ResponsesAPIResponse,
)
from app.request_transform import transform_request
from app.response_transform import transform_response
from app.stream_transform import stream_transform_iter
from app.vllm_client import vllm_client

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger("aigateway")

app = FastAPI(title="AIGateway", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    try:
        models = await vllm_client.models()
        return {"status": "ok", "vllm_models": [m.get("id") for m in models]}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


@app.post("/v1/responses")
async def create_response(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body"},
        )

    try:
        resp_request = ResponsesAPIRequest(**body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid request format: {e}"},
        )

    # Handle previous_response_id
    previous_messages = None
    if resp_request.previous_response_id:
        previous_messages = store.get(resp_request.previous_response_id)
        if previous_messages is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"previous_response_id '{resp_request.previous_response_id}' not found"
                },
            )

    chat_request = transform_request(resp_request, previous_messages)

    if resp_request.stream:
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        return StreamingResponse(
            _stream_response(chat_request, response_id, resp_request.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    chat_response = await vllm_client.complete(chat_request)
    responses_response = transform_response(chat_response)

    # Store conversation
    store.save(responses_response.id, chat_request.messages)

    return JSONResponse(content=responses_response.model_dump(exclude_none=True))


async def _stream_response(chat_request, response_id: str, model: str):
    try:
        line_iter = vllm_client.stream(chat_request)
        async for event in stream_transform_iter(line_iter, response_id, model):
            yield event
    except Exception as e:
        logger.error("Stream error: %s", e)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        # Emit a completed event so the client doesn't hang waiting for it.
        yield _sse_line(
            "response.completed",
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "model": model,
                    "status": "failed",
                    "output": [],
                    "error": str(e),
                },
            },
        )
        yield 'event: response.done\ndata: {"type": "response.done"}\n\n'


def _sse_line(event_type: str, data: dict) -> str:
    # Codex CLI requires a "type" field in each SSE data payload for serde.
    payload = {**data, "type": event_type}
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


@app.on_event("shutdown")
async def shutdown():
    await vllm_client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.proxy_port)