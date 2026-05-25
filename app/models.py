from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ─── Responses API Request Models ───

class InputText(BaseModel):
    type: Literal["input_text", "output_text"] = "input_text"
    text: str


class InputImage(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: str | None = None


class InputFile(BaseModel):
    type: Literal["input_file"] = "input_file"
    file_url: str | None = None
    file_data: str | None = None
    filename: str | None = None


InputContent = InputText | InputImage | InputFile


class ResponseInputMessage(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "developer", "system"]
    content: str | list[InputContent]


class FunctionCallResult(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class FunctionCallInput(BaseModel):
    type: Literal["function_call"] = "function_call"
    name: str
    call_id: str
    arguments: str


ResponseInputItem = ResponseInputMessage | FunctionCallResult | FunctionCallInput


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class TextFormat(BaseModel):
    type: Literal["json_schema", "json_object", "text"] = "text"
    json_schema: dict[str, Any] | None = None


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class ResponsesAPIRequest(BaseModel):
    model: str
    input: str | list[ResponseInputItem] = Field(default_factory=list)
    instructions: str | None = None
    tools: list[FunctionTool | Any] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    text: TextFormat | None = None
    reasoning: ReasoningConfig | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    metadata: dict[str, Any] | None = None


# ─── Chat Completions API Models ───

class ChatToolCallFunction(BaseModel):
    name: str
    arguments: str


class ChatToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ChatToolCallFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatFunctionTool(BaseModel):
    type: Literal["function"] = "function"
    function: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[ChatFunctionTool] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    extra_body: dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None
    model: str | None = None


# ─── Chat Completions Streaming Models ───

class ChatStreamToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ChatStreamToolCall(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: ChatStreamToolCallFunction


class ChatCompletionStreamDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ChatStreamToolCall] | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    choices: list[ChatCompletionStreamChoice]
    usage: ChatCompletionUsage | None = None
    model: str | None = None


# ─── Responses API Response Models ───

class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseOutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str | None = None
    role: Literal["assistant"] = "assistant"
    content: list[OutputText]


class ResponseFunctionCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str


ResponseOutputItem = ResponseOutputMessage | ResponseFunctionCall


class ResponsesAPIUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesAPIResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    model: str | None = None
    output: list[ResponseOutputItem]
    status: Literal["completed", "incomplete", "failed"] = "completed"
    usage: ResponsesAPIUsage | None = None
    metadata: dict[str, Any] | None = None