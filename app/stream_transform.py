from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from app.models import ChatCompletionStreamChunk, ChatMessage, ChatToolCall, ChatToolCallFunction

logger = logging.getLogger("aigateway")


class StreamState:
    def __init__(self, response_id: str, model: str | None = None):
        self.response_id = response_id
        self.model = model
        self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.text_accumulator = ""
        self.tool_call_accumulators: dict[int, dict[str, str]] = {}
        self.message_item_started = False
        self.text_started = False
        self.initial_events_emitted = False
        self.finished = False
        self.created_at = int(time.time())

    @property
    def message_offset(self) -> int:
        return 1 if self.message_item_started else 0


def emit_initial_events(state: StreamState) -> list[str]:
    if state.initial_events_emitted:
        return []
    state.initial_events_emitted = True

    return [
        _sse_line(
            "response.created",
            {
                "response": {
                    "id": state.response_id,
                    "object": "response",
                    "model": state.model,
                    "status": "in_progress",
                    "output": [],
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "created_at": state.created_at,
                },
            },
        )
    ]


def _emit_message_start(state: StreamState) -> list[str]:
    if state.message_item_started:
        return []
    state.message_item_started = True

    return [
        _sse_line(
            "response.output_item.added",
            {
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": state.msg_id,
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
    ]


def process_chunk(chunk: ChatCompletionStreamChunk, state: StreamState) -> list[str]:
    lines = emit_initial_events(state)

    if not chunk.choices:
        return lines

    choice = chunk.choices[0]
    delta = choice.delta

    if delta.content:
        if not state.text_started:
            state.text_started = True
            lines.extend(_emit_message_start(state))
        state.text_accumulator += delta.content
        lines.append(
            _sse_line(
                "response.output_text.delta",
                {
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta.content,
                },
            )
        )

    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in state.tool_call_accumulators:
                call_id = tc.id or f"call_{uuid.uuid4().hex[:24]}"
                state.tool_call_accumulators[idx] = {
                    "id": call_id,
                    "name": tc.function.name or "",
                    "arguments": "",
                    "fc_id": f"fc_{uuid.uuid4().hex[:24]}",
                }
                output_index = idx + state.message_offset
                lines.append(
                    _sse_line(
                        "response.output_item.added",
                        {
                            "output_index": output_index,
                            "item": {
                                "type": "function_call",
                                "id": state.tool_call_accumulators[idx]["fc_id"],
                                "call_id": call_id,
                                "name": tc.function.name or "",
                                "arguments": "",
                                "status": "in_progress",
                            },
                        },
                    )
                )

            if tc.function.name:
                state.tool_call_accumulators[idx]["name"] = tc.function.name
            if tc.function.arguments:
                state.tool_call_accumulators[idx]["arguments"] += tc.function.arguments
                output_index = idx + state.message_offset
                lines.append(
                    _sse_line(
                        "response.function_call_arguments.delta",
                        {
                            "output_index": output_index,
                            "item_id": state.tool_call_accumulators[idx]["fc_id"],
                            "delta": tc.function.arguments,
                        },
                    )
                )

    if choice.finish_reason:
        lines.extend(emit_done_events(state, chunk))

    return lines


def emit_done_events(state: StreamState, chunk: ChatCompletionStreamChunk) -> list[str]:
    if state.finished:
        return []
    state.finished = True

    lines = []

    if state.message_item_started:
        if state.text_started:
            lines.append(
                _sse_line(
                    "response.output_text.done",
                    {
                        "output_index": 0,
                        "content_index": 0,
                        "text": state.text_accumulator,
                    },
                )
            )
            lines.append(
                _sse_line(
                    "response.content_part.done",
                    {
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": state.text_accumulator,
                        },
                    },
                )
            )

        content = []
        if state.text_accumulator:
            content.append({"type": "output_text", "text": state.text_accumulator})
        lines.append(
            _sse_line(
                "response.output_item.done",
                {
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": state.msg_id,
                        "role": "assistant",
                        "content": content,
                    },
                },
            )
        )

    for idx, acc in sorted(state.tool_call_accumulators.items()):
        output_index = idx + state.message_offset
        lines.append(
            _sse_line(
                "response.function_call_arguments.done",
                {
                    "output_index": output_index,
                    "item_id": acc["fc_id"],
                    "arguments": acc["arguments"],
                },
            )
        )
        lines.append(
            _sse_line(
                "response.output_item.done",
                {
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": acc.get("fc_id", ""),
                        "call_id": acc["id"],
                        "name": acc["name"],
                        "arguments": acc["arguments"],
                        "status": "completed",
                    },
                },
            )
        )

    output_items = []
    if state.text_accumulator:
        output_items.append(
            {
                "type": "message",
                "id": state.msg_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": state.text_accumulator}],
            }
        )
    for idx, acc in sorted(state.tool_call_accumulators.items()):
        output_items.append(
            {
                "type": "function_call",
                "id": acc.get("fc_id", ""),
                "call_id": acc["id"],
                "name": acc["name"],
                "arguments": acc["arguments"],
            }
        )

    finish_reason = chunk.choices[0].finish_reason if chunk.choices else "stop"
    status = "completed" if finish_reason in ("stop", "tool_calls") else "incomplete"
    end_turn = finish_reason == "stop"

    # Always include usage as a non-null object to avoid nil pointer
    # dereference in upstream gateways (one-api / new-api).
    usage_data = {
        "input_tokens": chunk.usage.prompt_tokens if chunk.usage else 0,
        "output_tokens": chunk.usage.completion_tokens if chunk.usage else 0,
        "total_tokens": chunk.usage.total_tokens if chunk.usage else 0,
    }

    lines.append(
        _sse_line(
            "response.completed",
            {
                "response": {
                    "id": state.response_id,
                    "object": "response",
                    "model": state.model,
                    "status": status,
                    "output": output_items,
                    "usage": usage_data,
                    "created_at": state.created_at,
                    "end_turn": end_turn,
                },
            },
        )
    )

    return lines


async def stream_transform_iter(
    chunk_iter,
    response_id: str,
    model: str | None = None,
    messages: list[ChatMessage] | None = None,
):
    state = StreamState(response_id, model)
    raw_line_count = 0
    chunk_count = 0
    logger.info("stream_transform_iter started: response_id=%s, model=%s", response_id, model)

    try:
        async for raw_line in chunk_iter:
            raw_line_count += 1
            line = (
                raw_line.strip()
                if isinstance(raw_line, str)
                else raw_line.decode("utf-8").strip()
            )
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                logger.info(
                    "Stream transform: [DONE] received after %d raw lines, %d chunks",
                    raw_line_count,
                    chunk_count,
                )
                break

            try:
                chunk_data = json.loads(data)
                chunk = ChatCompletionStreamChunk(**chunk_data)
                chunk_count += 1
            except json.JSONDecodeError:
                logger.warning("Skipping non-JSON SSE line: %s", data[:200])
                continue
            except Exception as exc:
                logger.warning("Skipping unparseable chunk: %s", exc)
                continue

            for event_line in process_chunk(chunk, state):
                logger.debug("Yielding event: %s", event_line[:150])
                yield event_line
    except Exception as exc:
        logger.error("Stream iterator error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    # Guarantee response.completed is always emitted, even if stream
    # ended without [DONE] or finish_reason.
    if not state.finished:
        logger.warning(
            "Stream ended without finish_reason; emitting synthetic completion "
            "(raw_lines=%d, chunks=%d)",
            raw_line_count,
            chunk_count,
        )
        synthetic = ChatCompletionStreamChunk(id=response_id, choices=[])
        for event_line in emit_done_events(state, synthetic):
            yield event_line

    logger.info(
        "Stream transform complete: response.completed emitted, "
        "raw_lines=%d, chunks=%d, text_len=%d, tool_calls=%d",
        raw_line_count,
        chunk_count,
        len(state.text_accumulator),
        len(state.tool_call_accumulators),
    )

    # Warn when the model produces a text-only response without tool calls,
    # which likely means it summarized prematurely instead of continuing work.
    if state.text_accumulator and not state.tool_call_accumulators and state.finished:
        finish_reason = ""
        # We can't easily access the finish_reason here since state doesn't store it,
        # but the warning is still useful for diagnostics.
        logger.warning(
            "Model stopped with text-only response (no tool calls). "
            "This may indicate premature summary. text_len=%d",
            len(state.text_accumulator),
        )

    yield "event: response.done\ndata: {\"type\": \"response.done\"}\n\n"

    # Save conversation so follow-up requests with previous_response_id
    # can retrieve the message history.  System messages are NOT saved —
    # they are reconstructed from the current request each turn, which
    # prevents duplicate accumulation that causes the model to summarize
    # prematurely instead of continuing tool execution.
    if messages is not None:
        from app.conversation_store import store

        non_system = [m for m in messages if m.role != "system"]
        assistant_msg = _build_assistant_message(state)
        saved = non_system + [assistant_msg]
        store.save(response_id, saved)
        logger.info(
            "Saved conversation for response_id=%s, msg_count=%d (saved=%d, system_filtered=%d), has_tool_calls=%s",
            response_id,
            len(messages),
            len(saved),
            len(messages) - len(non_system),
            bool(state.tool_call_accumulators),
        )


def _sse_line(event_type: str, data: dict[str, Any]) -> str:
    # Codex CLI requires a "type" field in each SSE data payload
    # for serde deserialization.  It must match the event type string.
    payload = {**data, "type": event_type}
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _build_assistant_message(state: StreamState) -> ChatMessage:
    tool_calls = None
    if state.tool_call_accumulators:
        tool_calls = []
        for idx in sorted(state.tool_call_accumulators):
            acc = state.tool_call_accumulators[idx]
            tool_calls.append(
                ChatToolCall(
                    id=acc["id"],
                    function=ChatToolCallFunction(
                        name=acc["name"],
                        arguments=acc["arguments"],
                    ),
                )
            )

    content = state.text_accumulator or None
    return ChatMessage(role="assistant", content=content, tool_calls=tool_calls)