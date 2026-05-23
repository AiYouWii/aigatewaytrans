from __future__ import annotations

import uuid

from app.models import (
    ChatCompletionResponse,
    OutputText,
    ResponseFunctionCall,
    ResponseOutputMessage,
    ResponsesAPIResponse,
    ResponsesAPIUsage,
)


def transform_response(
    chat_response: ChatCompletionResponse,
) -> ResponsesAPIResponse:
    output_items = []

    for choice in chat_response.choices:
        message = choice.message

        if message.content:
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            output_items.append(
                ResponseOutputMessage(
                    type="message",
                    id=msg_id,
                    role="assistant",
                    content=[OutputText(type="output_text", text=message.content)],
                )
            )

        if message.tool_calls:
            for tc in message.tool_calls:
                output_items.append(
                    ResponseFunctionCall(
                        type="function_call",
                        id=f"fc_{uuid.uuid4().hex[:24]}",
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )

    status = _map_finish_reason(
        chat_response.choices[0].finish_reason if chat_response.choices else None
    )

    usage = None
    if chat_response.usage:
        usage = ResponsesAPIUsage(
            input_tokens=chat_response.usage.prompt_tokens,
            output_tokens=chat_response.usage.completion_tokens,
            total_tokens=chat_response.usage.total_tokens,
        )

    return ResponsesAPIResponse(
        id=f"resp_{uuid.uuid4().hex[:24]}",
        model=chat_response.model,
        output=output_items,
        status=status,
        usage=usage,
    )


def _map_finish_reason(reason: str | None) -> str:
    mapping = {
        "stop": "completed",
        "length": "incomplete",
        "tool_calls": "completed",
        "content_filter": "incomplete",
    }
    return mapping.get(reason, "completed")