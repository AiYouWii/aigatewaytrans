from __future__ import annotations

import logging

from app.config import settings
from app.models import (
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatToolCall,
    ChatToolCallFunction,
    FunctionCallInput,
    FunctionCallResult,
    FunctionTool,
    InputText,
    ResponsesAPIRequest,
    ResponseInputMessage,
)

logger = logging.getLogger("aigateway")

_TOOL_CONTINUATION_HINT = (
    "IMPORTANT: When tools are available, you must continue executing them "
    "step by step until all requested changes are fully completed. Do NOT "
    "stop early to provide a summary. Only output a final text response "
    "after every modification has been applied."
)

_TOOL_RESULT_HINT = (
    "\n\n[Continue with the next tool call if more work remains "
    "to complete the task. Do not provide a summary prematurely.]"
)


def _trim_context(messages: list[ChatMessage], max_non_system: int) -> list[ChatMessage]:
    """Trim long conversations to keep the model focused.

    Keeps: system message + first user message + most recent messages.
    Drops older middle messages that are no longer essential.
    Ensures the trimmed conversation starts at a valid boundary
    (a user message, or an assistant message with tool_calls before a tool message).
    """
    if len(messages) <= max_non_system + 1:  # +1 for system
        return messages

    system = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]

    if len(non_system) <= max_non_system:
        return messages

    # Keep the first user message (original task description)
    first_user = None
    rest_start = 0
    for i, m in enumerate(non_system):
        if m.role == "user":
            first_user = m
            rest_start = i + 1
            break

    rest = non_system[rest_start:]

    # From the rest, keep the most recent messages up to our budget
    budget = max_non_system - (1 if first_user else 0)

    # Walk forward from the cut point to find a valid conversation boundary:
    # - a user message (start of a new turn)
    # - an assistant message with tool_calls (preceding a tool message)
    # This ensures tool messages always have a matching assistant message.
    cut = len(rest) - budget
    for i in range(max(cut, 0), len(rest)):
        m = rest[i]
        if m.role == "user":
            cut = i
            break
        if m.role == "assistant" and m.tool_calls:
            cut = i
            break

    recent = rest[cut:]

    dropped = len(rest) - len(recent)
    logger.info(
        "Context trimmed: %d non-system messages → %d (dropped %d older messages)",
        len(non_system),
        max_non_system,
        dropped,
    )

    result = system
    if first_user:
        result.append(first_user)
    result.extend(recent)
    return result


def transform_request(
    request: ResponsesAPIRequest,
    previous_messages: list[ChatMessage] | None = None,
    max_output_tokens_default: int | None = None,
    max_context_messages_limit: int | None = None,
) -> ChatCompletionRequest:
    raw_messages = []

    if request.instructions:
        raw_messages.append(ChatMessage(role="system", content=request.instructions))

    # Inject continuation instructions when tools are available, to prevent
    # the model from stopping early with a summary instead of making tool calls.
    # Since system messages are filtered from saved conversations (not duplicated),
    # this hint is only injected once per turn.
    has_tools = request.tools is not None and len(request.tools) > 0
    if has_tools:
        raw_messages.append(ChatMessage(role="system", content=_TOOL_CONTINUATION_HINT))

    if previous_messages:
        raw_messages.extend(previous_messages)

    # Collect existing call_ids from previous_messages to avoid
    # duplicate assistant messages when Codex re-includes function_call
    # items that are already in the conversation history.
    existing_call_ids: set[str] = set()
    if previous_messages:
        for msg in previous_messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    existing_call_ids.add(tc.id)

    if isinstance(request.input, str):
        raw_messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        for item in request.input:
            msg = _transform_input_item(item, existing_call_ids, has_tools)
            if msg:
                raw_messages.append(msg)

    # vLLM requires a single system message at the beginning.
    # Merge all system-role messages into one, then keep non-system messages in order.
    system_parts: list[str] = []
    non_system: list[ChatMessage] = []
    for m in raw_messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
        else:
            non_system.append(m)

    messages: list[ChatMessage] = []
    if system_parts:
        messages.append(ChatMessage(role="system", content="\n\n".join(system_parts)))
    messages.extend(non_system)

    # Trim long context to prevent the model from getting confused
    messages = _trim_context(messages, max_context_messages_limit or settings.fallback_max_context_messages)

    tools = _transform_tools(request.tools)

    response_format = None
    if request.text and request.text.type != "text":
        if request.text.type == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": request.text.json_schema,
            }
        elif request.text.type == "json_object":
            response_format = {"type": "json_object"}

    reasoning_effort = None
    if request.reasoning and request.reasoning.effort:
        reasoning_effort = request.reasoning.effort

    # Use dynamic max_output_tokens when client doesn't specify one,
    # giving the model enough room for full tool-call responses.
    max_tokens = request.max_output_tokens or (max_output_tokens_default or settings.fallback_max_output_tokens)

    logger.info(
        "Transformed request: model=%s, msg_count=%d, has_tools=%s, stream=%s, max_tokens=%s",
        request.model,
        len(messages),
        tools is not None,
        request.stream,
        max_tokens,
    )

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        tools=tools,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=max_tokens,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )


def _transform_input_item(
    item: dict | object,
    existing_call_ids: set[str] | None = None,
    has_tools: bool = False,
) -> ChatMessage | None:
    if isinstance(item, dict):
        item_type = item.get("type")
    else:
        item_type = getattr(item, "type", None)

    if item_type == "message":
        if isinstance(item, dict):
            msg = ResponseInputMessage(**item)
        else:
            msg = item
        return _transform_message(msg)

    if item_type == "function_call_output":
        if isinstance(item, dict):
            result = FunctionCallResult(**item)
        else:
            result = item
        content = result.output
        # Append continuation hint to tool results when tools are available,
        # nudging the model to keep executing instead of summarizing prematurely.
        if has_tools:
            content += _TOOL_RESULT_HINT
        return ChatMessage(
            role="tool",
            tool_call_id=result.call_id,
            content=content,
        )

    if item_type == "function_call":
        if isinstance(item, dict):
            call = FunctionCallInput(**item)
        else:
            call = item
        # Skip duplicate function_calls already present in previous_messages
        if existing_call_ids and call.call_id in existing_call_ids:
            return None
        return ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatToolCall(
                    id=call.call_id,
                    function=ChatToolCallFunction(
                        name=call.name,
                        arguments=call.arguments,
                    ),
                )
            ],
        )

    return None


def _transform_message(msg: ResponseInputMessage) -> ChatMessage:
    role = _map_role(msg.role)

    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        parts = []
        for part in msg.content:
            if isinstance(part, dict):
                if part.get("type") in ("input_text", "output_text"):
                    parts.append(part.get("text", ""))
                elif part.get("type") == "input_image":
                    parts.append(part.get("image_url", ""))
            elif isinstance(part, InputText):
                parts.append(part.text)
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
        content = "\n".join(parts)
    else:
        content = str(msg.content)

    return ChatMessage(role=role, content=content)


def _map_role(role: str) -> str:
    mapping = {
        "developer": "system",
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }
    return mapping.get(role, role)


def _transform_tools(
    tools: list[FunctionTool | dict] | None,
) -> list[ChatFunctionTool] | None:
    if not tools:
        return None

    result = []
    for tool in tools:
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            if tool_type == "function":
                result.append(
                    ChatFunctionTool(
                        type="function",
                        function={
                            "name": tool.get("name"),
                            "description": tool.get("description"),
                            "parameters": tool.get("parameters"),
                        },
                    )
                )
            else:
                continue
        elif isinstance(tool, FunctionTool):
            result.append(
                ChatFunctionTool(
                    type="function",
                    function={
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                )
            )
        else:
            tool_dict = tool if isinstance(tool, dict) else {}
            if tool_dict.get("type") == "function":
                result.append(
                    ChatFunctionTool(
                        type="function",
                        function=tool_dict,
                    )
                )

    return result if result else None