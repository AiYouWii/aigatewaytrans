from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

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

def _setup_logging() -> logging.Logger:
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    level = getattr(logging, settings.log_level, logging.INFO)

    app_logger = logging.getLogger("aigateway")
    app_logger.setLevel(level)

    # Console handler (always active)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    app_logger.addHandler(stream_handler)

    # File handler (when log_file is configured)
    if settings.log_file:
        log_dir = os.path.dirname(settings.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file,
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backup_count,
        )
        file_handler.setFormatter(formatter)
        app_logger.addHandler(file_handler)

    # Fallback basicConfig for uvicorn/access logs
    logging.basicConfig(level=level, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    return app_logger


logger = _setup_logging()

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
            logger.warning(
                "previous_response_id '%s' not found in store",
                resp_request.previous_response_id,
            )
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"previous_response_id '{resp_request.previous_response_id}' not found"
                },
            )
        logger.info(
            "Continuing conversation: previous_response_id=%s, "
            "prev_msg_count=%d, new_input_items=%d",
            resp_request.previous_response_id,
            len(previous_messages),
            len(resp_request.input) if isinstance(resp_request.input, list) else 1,
        )

    # Fetch model config from vLLM to dynamically adapt parameters
    max_model_len = None
    try:
        model_config = await vllm_client.get_model_config(resp_request.model)
        max_model_len = model_config.get("max_model_len")
    except Exception:
        logger.warning("Cannot fetch model config, using fallback defaults")

    # Compute dynamic limits based on model's actual context window:
    # - max_output_tokens: reserve ~25% of context for model output
    # - max_context_messages: estimate ~400 tokens per message,
    #   subtract output budget from total context
    if max_model_len:
        dyn_max_output_tokens = min(max_model_len // 4, settings.fallback_max_output_tokens)
        dyn_max_context = min(max(max_model_len - dyn_max_output_tokens, 8000) // 400, 100)
    else:
        dyn_max_output_tokens = settings.fallback_max_output_tokens
        dyn_max_context = settings.fallback_max_context_messages

    chat_request = transform_request(
        resp_request,
        previous_messages,
        max_output_tokens_default=dyn_max_output_tokens,
        max_context_messages_limit=dyn_max_context,
    )

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

    # Store conversation — exclude system messages to prevent
    # duplicate accumulation on follow-up turns.
    non_system = [m for m in chat_request.messages if m.role != "system"]
    store.save(responses_response.id, non_system)

    return JSONResponse(content=responses_response.model_dump(exclude_none=True))


async def _stream_response(chat_request, response_id: str, model: str):
    logger.info("Stream response started: response_id=%s, model=%s", response_id, model)
    created_at = int(time.time())
    event_count = 0
    try:
        line_iter = vllm_client.stream(chat_request)
        async for event in stream_transform_iter(
            line_iter, response_id, model, chat_request.messages
        ):
            event_count += 1
            if event_count <= 5 or event_count % 20 == 0:
                logger.debug("Stream event #%d: %s", event_count, event[:120] if isinstance(event, str) else str(event)[:120])
            yield event
    except Exception as e:
        logger.error("Stream error: %s", e)
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        yield _sse_line(
            "response.completed",
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "model": model,
                    "status": "failed",
                    "output": [],
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "created_at": created_at,
                    "end_turn": False,
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


# ─── Catch-all proxy: forward unmatched requests to vLLM ───

_PROXY_SKIP_HEADERS = {
    "content-encoding", "content-length", "transfer-encoding", "connection",
}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_to_vllm(request: Request, path: str):
    vllm_path = f"/{path}"
    logger.debug("Proxying %s %s → vLLM", request.method, vllm_path)

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _PROXY_SKIP_HEADERS
    }
    body = await request.body()

    resp = await vllm_client.proxy(request.method, vllm_path, headers, body)

    resp_headers = {}
    for key, value in resp.headers.items():
        if key.lower() not in _PROXY_SKIP_HEADERS:
            resp_headers[key] = value

    # SSE streams: relay the raw bytes without buffering.
    if resp_headers.get("content-type", "").startswith("text/event-stream"):
        return StreamingResponse(
            _relay_stream(resp),
            status_code=resp.status_code,
            headers=resp_headers,
            media_type="text/event-stream",
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp_headers.get("content-type"),
    )


async def _relay_stream(resp):
    async for chunk in resp.aiter_bytes():
        yield chunk


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.proxy_port)