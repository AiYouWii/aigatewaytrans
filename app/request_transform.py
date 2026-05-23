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


def transform_request(
    request: ResponsesAPIRequest,
    previous_messages: list[ChatMessage] | None = None,
) -> ChatCompletionRequest:
    messages = []

    if previous_messages:
        messages.extend(previous_messages)

    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))

    if isinstance(request.input, str):
        messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        for item in request.input:
            msg = _transform_input_item(item)
            if msg:
                messages.append(msg)

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


def _transform_input_item(item: dict | object) -> ChatMessage | None:
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
        return ChatMessage(
            role="tool",
            tool_call_id=result.call_id,
            content=result.output,
        )

    if item_type == "function_call":
        if isinstance(item, dict):
            call = FunctionCallInput(**item)
        else:
            call = item
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
                if part.get("type") == "input_text":
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