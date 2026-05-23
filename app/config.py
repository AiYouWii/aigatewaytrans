from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_base_url: str = "http://localhost:8000"
    proxy_port: int = 9000
    log_level: str = "INFO"

    model_config = {"env_prefix": "AIGATEWAY_", "env_file": ".env"}


settings = Settings()