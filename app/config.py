"""Settings loaded from environment variables or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which backend to talk to: "openai", "azure" or "ollama" (free, runs locally)
    llm_provider: Literal["openai", "azure", "ollama"] = "ollama"

    # Model or deployment name. For Azure this is your *deployment* name.
    chat_model: str = "llama3.2"

    # OpenAI
    openai_api_key: str | None = None

    # Azure OpenAI
    azure_openai_endpoint: str | None = None  # e.g. https://my-resource.openai.azure.com
    azure_openai_api_key: str | None = None
    # Leave empty to use Azure's OpenAI-compatible v1 endpoint.
    # Set it (e.g. "2024-10-21") only if your resource requires a dated API version.
    azure_openai_api_version: str | None = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"

    # Safety limits
    max_tool_steps: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
