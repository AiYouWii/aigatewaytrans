from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_base_url: str = "http://localhost:8000"
    proxy_port: int = 9000
    log_level: str = "INFO"
    log_file: str = ""
    log_file_max_bytes: int = 10_000_000
    log_file_backup_count: int = 5
    # Fallback defaults when vLLM model info is unavailable.
    # These are automatically computed from vLLM's max_model_len at runtime.
    fallback_max_output_tokens: int = 16384
    fallback_max_context_messages: int = 60

    model_config = {"env_prefix": "AIGATEWAY_", "env_file": ".env"}


settings = Settings()