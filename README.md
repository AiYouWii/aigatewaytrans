# AIGateway

OpenAI Responses API → Chat Completions API 代理，让 Codex CLI 等使用 Responses API 的工具可以连接 vLLM 部署的模型。

## 解决的问题

vLLM 仅支持 OpenAI Chat Completions 协议 (`/v1/chat/completions`)，而 Codex CLI 使用较新的 Responses API 协议 (`/v1/responses`)。直连会出现两类错误：

1. **`developer` role 不支持** — vLLM / Qwen 仅接受 `system` / `user` / `assistant` / `tool` 角色
2. **工具调用格式不匹配** — Responses API 和 Chat Completions API 的 tool 格式不同

本代理在两者之间做协议转换，使 Codex CLI 可以正常连接 vLLM。

## 快速开始

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env，确认 vLLM 地址和端口
```

`.env` 配置项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AIGATEWAY_VLLM_BASE_URL` | `http://localhost:8000` | vLLM 服务地址 |
| `AIGATEWAY_PROXY_PORT` | `9000` | 代理监听端口 |
| `AIGATEWAY_LOG_LEVEL` | `INFO` | 日志级别 |

### 2. 本地运行

```bash
pip install -r requirements.txt
python -m app.main
```

或使用启动脚本：

```bash
./run.sh
```

### 3. Docker 运行

```bash
docker build -t aigateway .
docker run -p 9000:9000 --env-file .env aigateway
```

### 4. Codex CLI 配置

将 Codex 的 API base URL 设为代理地址：

```bash
export OPENAI_BASE_URL=http://localhost:9000/v1
export OPENAI_API_KEY=your-key  # vLLM 不校验 key，随意填写
```

## 架构

```
Codex CLI ──→ AIGateway (:9000) ──→ vLLM (:8000)
              /v1/responses          /v1/chat/completions
              Responses API 格式     Chat Completions 格式
```

## 协议转换详解

### Request（Responses → Chat Completions）

| Responses API | Chat Completions | 转换规则 |
|---------------|-----------------|----------|
| `input[]` | `messages[]` | 整组转换 |
| `instructions` | 首条 `system` message | 顶层指令映射 |
| `role: "developer"` | `role: "system"` | 角色映射 |
| `content: [{type: "input_text"}]` | `content: string` | 内容扁平化 |
| `tools[{type:"function", name, ...}]` | `tools[{type:"function", function: {name, ...}}]` | 扁平 → 嵌套 |
| `function_call_output` | `role: "tool", tool_call_id, content` | 工具结果转换 |
| `function_call` (input) | `role: "assistant", tool_calls[]` | 工具调用回填 |
| `text.format` | `response_format` | 结构化输出映射 |
| `reasoning.effort` | `reasoning_effort` | 推理强度映射 |
| `max_output_tokens` | `max_tokens` | 参数名映射 |

### Response（Chat Completions → Responses）

| Chat Completions | Responses API | 转换规则 |
|------------------|---------------|----------|
| `choices[].message` | `output[]` (type: "message") | 输出项转换 |
| `content: string` | `content: [{type: "output_text"}]` | 内容包装 |
| `tool_calls[].function` | `{type: "function_call", call_id}` | 工具调用转换 |
| `finish_reason` | `status` | stop→completed, length→incomplete |

### Streaming（SSE）

Chat Completions 流式发送 `delta` 事件，代理逐条转换为 Responses API 的生命周期事件：

| 时机 | 事件类型 |
|------|----------|
| 开始 | `response.created` |
| 输出开始 | `response.output_item.added` |
| 内容开始 | `response.content_part.added` |
| 文本增量 | `response.output_text.delta` |
| 工具调用增量 | `response.function_call_arguments.delta` |
| 文本完成 | `response.output_text.done` |
| 输出项完成 | `response.output_item.done` |
| 整体完成 | `response.completed` |

流式处理是增量式的——不缓冲全部 chunk 后再输出，而是逐条转换即时推送。

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/responses` | POST | Responses API 代理（主要端点） |
| `/health` | GET | 健康检查，返回 vLLM 连接状态和可用模型列表 |

## 项目结构

```
app/
├── main.py              # FastAPI 入口，路由和中间件
├── config.py            # 环境变量配置
├── models.py            # 两端 API 的 Pydantic 模型
├── request_transform.py # Responses 请求 → Chat Completions 请求
├── response_transform.py# Chat Completions 响应 → Responses 响应
├── stream_transform.py  # SSE 流式事件转换
├── conversation_store.py# previous_response_id 上下文存储
├── vllm_client.py       # vLLM HTTP 客户端
```

## 不支持的 Responses API 特性

以下特性 vLLM 不具备对应能力，代理会静默忽略：

- `web_search` / `file_search` / `code_interpreter` / `computer` — 内置工具类型
- `store` / `include` — 响应存储与附加数据
- `max_tool_calls` / `truncation` — 控制参数
- `conversation` — 持久会话 ID（`previous_response_id` 通过内存存储有限支持）

## 测试示例

```bash
# 基础文本请求
curl -X POST http://localhost:9000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-6-35b-a3b",
    "input": [
      {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be concise"}]},
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
    ]
  }'

# 带工具调用的流式请求
curl -X POST http://localhost:9000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-6-35b-a3b",
    "stream": true,
    "input": [{"type": "message", "role": "user", "content": "What is the weather in NYC?"}],
    "tools": [{"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}]
  }'

# 健康检查
curl http://localhost:9000/health
```

## License

MIT