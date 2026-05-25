from __future__ import annotations

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


_TOOL_RESULT_HINT = (
    "\n\n[Continue with the next tool call if more work remains "
    "to complete the task. Do not provide a summary prematurely.]"
)


def transform_request(
    request: ResponsesAPIRequest,
    previous_messages: list[ChatMessage] | None = None,
) -> ChatCompletionRequest:
    raw_messages = []

    if request.instructions:
        raw_messages.append(ChatMessage(role="system", content=request.instructions))

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

    has_tools = request.tools is not None and len(request.tools) > 0

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

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        tools=tools,
        stream=request.stream,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
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