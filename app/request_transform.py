from __future__ import annotations

import logging

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


def transform_request(
    request: ResponsesAPIRequest,
    previous_messages: list[ChatMessage] | None = None,
    max_output_tokens_default: int | None = None,
    max_model_len: int | None = None,
) -> ChatCompletionRequest:
    raw_messages = []

    if request.instructions:
        raw_messages.append(ChatMessage(role="system", content=request.instructions))

    if previous_messages:
        raw_messages.extend(previous_messages)

    # Build call_id → function name mapping from conversation history.
    # This lets us add the `name` field to tool role messages, which is
    # how OpenAI formats them — giving the model better context about
    # which function produced each result.
    existing_call_ids: set[str] = set()
    call_id_to_name: dict[str, str] = {}
    if previous_messages:
        for msg in previous_messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    existing_call_ids.add(tc.id)
                    call_id_to_name[tc.id] = tc.function.name

    # Also collect call_id → name from function_call items in the
    # current input, so tool results that follow get the name too.
    if isinstance(request.input, list):
        for item in request.input:
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id_to_name[item.get("call_id", "")] = item.get("name", "")

    if isinstance(request.input, str):
        raw_messages.append(ChatMessage(role="user", content=request.input))
    elif isinstance(request.input, list):
        for item in request.input:
            msg = _transform_input_item(item, existing_call_ids, call_id_to_name)
            if msg:
                raw_messages.append(msg)

    # Merge consecutive assistant messages into single messages.
    # In the Responses API, a single assistant turn produces:
    #   - A message item (text content, role="assistant")
    #   - One or more function_call items (tool calls)
    # Our _transform_input_item creates separate ChatMessage for each,
    # resulting in multiple consecutive assistant messages. But Chat
    # Completions requires ALL content and tool_calls from one turn in
    # ONE assistant message. Without this merge, the conversation has
    # invalid structure (70 assistant messages instead of ~35), which
    # confuses the model and causes text-only responses.
    raw_messages = _merge_assistant_messages(raw_messages)

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

    # Compute max_tokens early — needed for truncation budget calculation.
    max_tokens = request.max_output_tokens or max_output_tokens_default

    # Auto-truncate context when it exceeds the model's limits.
    # OpenAI's Responses API defaults to truncation_strategy: "auto",
    # which silently drops old messages to keep the context manageable.
    # Without this, long tool-call chains overwhelm the model, causing
    # it to summarize with text instead of continuing tool execution.
    from app.config import settings
    if settings.auto_truncate and max_model_len and max_tokens:
        messages = _auto_truncate(messages, max_model_len, max_tokens)

    tools = _transform_tools(request.tools)

    # Translate tool_choice from Responses API format to Chat Completions format.
    # Responses API uses {"type": "function", "name": "func_name"}
    # Chat Completions uses {"type": "function", "function": {"name": "func_name"}}
    #
    # Codex sends tool_choice="auto" even when in a tool chain. Other models
    # (e.g. Qwen3) interpret "auto" as "you MAY respond with text instead of
    # calling tools", which causes premature text-only summaries that break
    # the tool chain. When the conversation already contains tool messages,
    # the agent is mid-execution — upgrade "auto" to "required" so the model
    # must continue calling tools.
    tool_choice = None
    has_tool_history = any(m.role == "tool" for m in raw_messages)

    if request.tool_choice:
        if isinstance(request.tool_choice, str):
            if request.tool_choice == "auto" and tools and has_tool_history:
                tool_choice = "required"
                logger.info("Upgraded tool_choice: auto → required (tool chain detected)")
            else:
                tool_choice = request.tool_choice
        elif isinstance(request.tool_choice, dict):
            if request.tool_choice.get("type") == "function":
                tool_choice = {
                    "type": "function",
                    "function": {"name": request.tool_choice.get("name")},
                }
            else:
                tool_choice = request.tool_choice
    elif tools:
        tool_choice = "required"

    # When the model is in a tool chain (tool_choice="required"),
    # reinforce the system message to prevent text-only summaries.
    # OpenAI's GPT-4o naturally continues tool chains; other models
    # (e.g. Qwen3) tend to summarize prematurely without this.
    if tool_choice == "required" and settings.reinforce_tool_use:
        _TOOL_USE_REINFORCEMENT = (
            "\n\nImportant: You are in the middle of executing a multi-step task. "
            "You must continue by calling the appropriate tool(s). "
            "Do NOT respond with only text or a summary. "
            "Always produce at least one tool call to continue the work."
        )
        for i, m in enumerate(messages):
            if m.role == "system":
                messages[i] = ChatMessage(
                    role="system",
                    content=m.content + _TOOL_USE_REINFORCEMENT,
                )
                logger.info("Reinforced system message for tool chain continuation")
                break

    parallel_tool_calls = request.parallel_tool_calls
    if parallel_tool_calls is None and tools:
        parallel_tool_calls = True

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

    logger.info(
        "Transformed request: model=%s, msg_count=%d, has_tools=%s, tool_choice=%s, parallel_tool_calls=%s, stream=%s, max_tokens=%s",
        request.model,
        len(messages),
        tools is not None,
        tool_choice,
        parallel_tool_calls,
        request.stream,
        max_tokens,
    )

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
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
    call_id_to_name: dict[str, str] | None = None,
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
        # Include `name` in tool messages — this is how OpenAI formats them
        # and helps the model understand which function the result belongs to.
        # Do NOT modify the content (no hints appended) — faithful translation.
        name = call_id_to_name.get(result.call_id) if call_id_to_name else None
        return ChatMessage(
            role="tool",
            tool_call_id=result.call_id,
            content=result.output,
            name=name,
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
                            "strict": tool.get("strict"),
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
                        "strict": tool.strict,
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


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    """Rough token estimate: ~4 chars per token + overhead for tool calls.

    Observed ratio between estimated and actual tokens is ~1.78x
    (estimated 64,840 vs actual 114,689). Apply a safety factor of
    1.8 to ensure auto_truncate kicks in BEFORE actual overflow.
    """
    total = 0
    for m in messages:
        if m.content:
            total += len(m.content) // 4 + 10  # content + role overhead
        if m.tool_calls:
            for tc in m.tool_calls:
                total += len(tc.function.arguments) // 4 + 20  # args + call metadata
        total += 4  # role/field overhead per message
    return int(total * 1.8)


def _auto_truncate(
    messages: list[ChatMessage],
    max_model_len: int,
    max_tokens: int,
) -> list[ChatMessage]:
    """Truncate old messages when context exceeds the model's limits.

    Matches OpenAI's default truncation_strategy: "auto" behavior.
    Keeps the system message and recent context, dropping old turns.
    A "turn" boundary is a position where we can safely cut without
    breaking conversation structure (before a user message, after
    tool results, or after an assistant text-only response).
    """
    budget = max_model_len - max_tokens
    if budget <= 0:
        budget = max_model_len // 2

    estimated = _estimate_tokens(messages)

    if estimated <= budget:
        return messages

    # Find safe truncation points in the non-system messages.
    # A safe point is before a user message, or at the start
    # (we always keep the system message).
    system_msg = messages[0] if messages and messages[0].role == "system" else None
    rest = messages[1:] if system_msg else messages

    # Identify turn boundaries: positions before user messages or
    # after tool/assistant exchanges where cutting won't break structure.
    boundaries: list[int] = []
    for i, m in enumerate(rest):
        if m.role == "user":
            boundaries.append(i)
        # After an assistant text-only response that follows tool
        # results — this completes a full exchange cycle.
        if i > 0 and m.role == "assistant" and not m.tool_calls:
            prev = rest[i - 1]
            if prev.role == "tool":
                boundaries.append(i + 1)

    if not boundaries:
        # No safe boundaries found; keep system + last few messages
        logger.warning(
            "No safe truncation boundaries found; keeping system + "
            "last 20 messages. original=%d, estimated_tokens=%d, budget=%d",
            len(messages),
            estimated,
            budget,
        )
        keep = rest[-20:] if len(rest) > 20 else rest
        return [system_msg] + keep if system_msg else keep

    # Try each boundary from the earliest, removing messages before it
    # until the estimated tokens fit within budget.
    for boundary in boundaries:
        truncated_rest = rest[boundary:]
        truncated = [system_msg] + truncated_rest if system_msg else truncated_rest
        new_estimated = _estimate_tokens(truncated)
        if new_estimated <= budget:
            removed = len(messages) - len(truncated)
            logger.info(
                "Auto-truncated: removed %d old messages, "
                "kept %d (estimated_tokens=%d→%d, budget=%d). "
                "This matches OpenAI's truncation_strategy:auto behavior.",
                removed,
                len(truncated),
                estimated,
                new_estimated,
                budget,
            )
            return truncated

    # If all boundaries still exceed budget, keep only system + last turn
    # (the most recent user/assistant/tool exchange).
    # Find the last user message and keep everything after it.
    last_user_idx = -1
    for i in range(len(rest) - 1, -1, -1):
        if rest[i].role == "user":
            last_user_idx = i
            break

    if last_user_idx >= 0:
        truncated_rest = rest[last_user_idx:]
        truncated = [system_msg] + truncated_rest if system_msg else truncated_rest
        removed = len(messages) - len(truncated)
        logger.warning(
            "Aggressive truncation: keeping only system + last turn "
            "(%d messages, removed=%d). estimated_tokens=%d, budget=%d",
            len(truncated),
            removed,
            _estimate_tokens(truncated),
            budget,
        )
        return truncated

    # Fallback: keep system + last 10 messages
    keep = rest[-10:] if len(rest) > 10 else rest
    truncated = [system_msg] + keep if system_msg else keep
    logger.warning(
        "Fallback truncation: keeping system + last 10 messages. "
        "original=%d, estimated_tokens=%d, budget=%d",
        len(messages),
        estimated,
        budget,
    )
    return truncated


def _merge_assistant_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Merge consecutive assistant messages into single messages.

    In the Responses API, a single assistant turn produces:
    - A message item (role="assistant", text content)
    - One or more function_call items (tool calls)

    Our _transform_input_item creates a separate ChatMessage for each,
    resulting in multiple consecutive assistant messages. But the Chat
    Completions format requires ALL content and tool_calls from one
    assistant turn to be in ONE message. Without this merge, the model
    sees invalid structure (e.g. 70 assistant msgs instead of ~35),
    which breaks the conversation format and causes text-only responses.
    """
    if not messages:
        return messages

    result: list[ChatMessage] = []
    i = 0
    merged_count = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role == "assistant":
            # Start collecting consecutive assistant messages
            parts_content: list[str] = []
            all_tool_calls: list[ChatToolCall] = []

            if msg.content:
                parts_content.append(msg.content)
            if msg.tool_calls:
                all_tool_calls.extend(msg.tool_calls)

            j = i + 1
            while j < len(messages) and messages[j].role == "assistant":
                next_msg = messages[j]
                if next_msg.content:
                    parts_content.append(next_msg.content)
                if next_msg.tool_calls:
                    all_tool_calls.extend(next_msg.tool_calls)
                j += 1

            if j > i + 1:
                merged_count += j - i - 1

            combined_content = "\n".join(parts_content) if parts_content else None
            combined_tool_calls = all_tool_calls if all_tool_calls else None

            result.append(ChatMessage(
                role="assistant",
                content=combined_content,
                tool_calls=combined_tool_calls,
            ))
            i = j
        else:
            result.append(msg)
            i += 1

    if merged_count > 0:
        logger.info(
            "Merged %d consecutive assistant messages into proper turns "
            "(%d → %d total messages)",
            merged_count,
            len(messages),
            len(result),
        )

    return result