from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_base_url: str = "http://localhost:8000"
    proxy_port: int = 9000
    log_level: str = "INFO"
    log_file: str = ""
    log_file_max_bytes: int = 10_000_000
    log_file_backup_count: int = 5
    # Fallback default when vLLM model info is unavailable.
    # This is automatically computed from vLLM's max_model_len at runtime.
    fallback_max_output_tokens: int = 16384
    # Auto-truncate context when it exceeds the model's context window.
    # Matches OpenAI's default truncation_strategy: "auto" behavior.
    # Set to False to disable (not recommended for long tool-call chains).
    auto_truncate: bool = True
    # When the model is in a tool chain (received a tool result), reinforce
    # the system message with a brief instruction to continue using tools.
    # This prevents text-only summaries that break the tool chain.
    # OpenAI's GPT-4o naturally continues tool chains; other models need
    # explicit instruction. Set to False for pure faithful translation.
    reinforce_tool_use: bool = True

    model_config = {"env_prefix": "AIGATEWAY_", "env_file": ".env"}


settings = Settings()