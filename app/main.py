from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
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

    logger.info(
        "Raw request: model=%s, tool_choice=%s, parallel_tool_calls=%s, "
        "has_tools=%s, stream=%s, max_output_tokens=%s, "
        "previous_response_id=%s, input_type=%s, input_items=%d, "
        "instructions_len=%d",
        resp_request.model,
        resp_request.tool_choice,
        resp_request.parallel_tool_calls,
        resp_request.tools is not None and len(resp_request.tools) > 0,
        resp_request.stream,
        resp_request.max_output_tokens,
        resp_request.previous_response_id,
        "str" if isinstance(resp_request.input, str) else "list",
        len(resp_request.input) if isinstance(resp_request.input, list) else 1,
        len(resp_request.instructions) if resp_request.instructions else 0,
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

    # Use max_output_tokens from the client if set; otherwise use the
    # dynamically computed default (from vLLM's max_model_len) to ensure
    # the model has enough room for full tool-call responses.
    max_model_len = None
    try:
        model_config = await vllm_client.get_model_config(resp_request.model)
        max_model_len = model_config.get("max_model_len")
    except Exception:
        logger.warning("Cannot fetch model config, using fallback defaults")

    if max_model_len:
        dyn_max_output_tokens = min(max_model_len // 4, settings.fallback_max_output_tokens)
    else:
        dyn_max_output_tokens = settings.fallback_max_output_tokens

    chat_request = transform_request(
        resp_request,
        previous_messages,
        max_output_tokens_default=dyn_max_output_tokens,
        max_model_len=max_model_len,
    )

    # Diagnostic: log message breakdown by role and estimated context size
    role_counts = {}
    for m in chat_request.messages:
        role_counts[m.role] = role_counts.get(m.role, 0) + 1
    estimated_tokens = sum(
        len(m.content or "") // 4
        + (len(m.tool_calls or []) * 100 if m.tool_calls else 0)
        for m in chat_request.messages
    )
    logger.info(
        "Request context: msg_count=%d, roles=%s, estimated_tokens=%d, "
        "max_model_len=%s, max_tokens=%s",
        len(chat_request.messages),
        role_counts,
        estimated_tokens,
        max_model_len,
        chat_request.max_tokens,
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

    try:
        chat_response = await vllm_client.complete(chat_request)
    except Exception as exc:
        error_text = str(exc)
        # Handle vLLM context overflow — return actionable error to agent
        if "maximum context length" in error_text or "context overflow" in error_text.lower():
            logger.warning("vLLM context overflow (non-streaming): %s", error_text[:300])
            return JSONResponse(
                status_code=400,
                content=_build_overflow_error_response(error_text, resp_request.model),
            )
        raise

    responses_response = transform_response(chat_response)

    # Store conversation — exclude system messages to prevent
    # duplicate accumulation on follow-up turns. Include the
    # assistant response so follow-up requests have full context.
    non_system = [m for m in chat_request.messages if m.role != "system"]
    assistant_msg = chat_response.choices[0].message
    if assistant_msg:
        saved = non_system + [assistant_msg]
    else:
        saved = non_system
    store.save(responses_response.id, saved)

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
        error_text = str(e)
        # Handle vLLM context overflow — emit actionable error as assistant text
        # with end_turn=False so the agent can compress and retry.
        if "maximum context length" in error_text or "context overflow" in error_text.lower():
            logger.warning("vLLM context overflow (streaming): %s", error_text[:300])
            for event in _build_overflow_sse_events(response_id, model, error_text):
                yield event
            yield 'event: response.done\ndata: {"type": "response.done"}\n\n'
            return

        # Other stream errors — fallback format
        logger.error("Stream error: %s", e)
        yield f"event: error\ndata: {json.dumps({'error': error_text})}\n\n"
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


def _parse_context_overflow(error_text: str) -> dict | None:
    """Parse vLLM context overflow error to extract token counts.

    vLLM returns messages like:
    "This model's maximum context length is 131072 tokens...
     your prompt contains at least 114689 input tokens and 16384
     requested output tokens, for a total of at least 131073 tokens."
    """
    max_match = re.search(r"maximum context length is (\d+)", error_text)
    input_match = re.search(r"at least (\d+) input tokens", error_text)
    output_match = re.search(r"(\d+) requested output tokens", error_text)
    total_match = re.search(r"total of at least (\d+) tokens", error_text)

    if max_match and (input_match or total_match):
        return {
            "max_context": int(max_match.group(1)),
            "input_tokens": int(input_match.group(1)) if input_match else None,
            "requested_output_tokens": int(output_match.group(1)) if output_match else None,
            "total_tokens": int(total_match.group(1)) if total_match else None,
        }
    return None


def _build_overflow_error_response(error_text: str, model: str) -> dict:
    """Build a Responses API error response for context overflow.

    Returns actionable information so the agent can compress or retry.
    """
    parsed = _parse_context_overflow(error_text)

    if parsed:
        return {
            "error": {
                "type": "context_overflow",
                "message": (
                    f"Context exceeds model limit: "
                    f"{parsed['input_tokens']} input tokens + "
                    f"{parsed['requested_output_tokens']} requested output tokens = "
                    f"{parsed['total_tokens']} total > "
                    f"{parsed['max_context']} max context. "
                    f"Please compress the conversation context or reduce output length and retry."
                ),
                "max_context_tokens": parsed["max_context"],
                "input_tokens": parsed["input_tokens"],
                "requested_output_tokens": parsed["requested_output_tokens"],
                "total_tokens": parsed["total_tokens"],
                "suggestion": "compress_context_or_retry",
            },
        }
    else:
        return {
            "error": {
                "type": "context_overflow",
                "message": f"Context overflow error from backend: {error_text[:500]}. Please compress the conversation context or reduce output length and retry.",
                "suggestion": "compress_context_or_retry",
            },
        }


def _build_overflow_sse_events(response_id: str, model: str, error_text: str) -> list[str]:
    """Build SSE events for context overflow error in streaming mode.

    Codex CLI expects: response.created → response.output_item.added →
    response.content_part.added → output_text.delta → ... → response.completed
    For an error, we emit a minimal valid stream with the error message
    as assistant text content, then response.completed with end_turn=False
    so Codex can see the error and decide to compress/retry.
    """
    error_data = _build_overflow_error_response(error_text, model)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    error_msg = error_data["error"]["message"]

    events = [
        _sse_line(
            "response.created",
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "model": model,
                    "status": "in_progress",
                    "output": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "created_at": created_at,
                },
            },
        ),
        _sse_line(
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "content": [],
                },
            },
        ),
        _sse_line(
            "response.content_part.added",
            {
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        _sse_line(
            "response.output_text.delta",
            {
                "output_index": 0,
                "content_index": 0,
                "delta": error_msg,
            },
        ),
        _sse_line(
            "response.output_text.done",
            {
                "output_index": 0,
                "content_index": 0,
                "text": error_msg,
            },
        ),
        _sse_line(
            "response.content_part.done",
            {
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": error_msg},
            },
        ),
        _sse_line(
            "response.output_item.done",
            {
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": error_msg}],
                },
            },
        ),
        _sse_line(
            "response.completed",
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "model": model,
                    "status": "incomplete",
                    "output": [
                        {
                            "type": "message",
                            "id": msg_id,
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": error_msg}],
                        },
                    ],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "created_at": created_at,
                    "end_turn": False,
                    "error": error_data["error"],
                },
            },
        ),
    ]

    return events


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