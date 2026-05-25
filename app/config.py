from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_base_url: str = "http://localhost:8000"
    proxy_port: int = 9000
    log_level: str = "INFO"
    log_file: str = ""
    log_file_max_bytes: int = 10_000_000
    log_file_backup_count: int = 5

    model_config = {"env_prefix": "AIGATEWAY_", "env_file": ".env"}


settings = Settings()