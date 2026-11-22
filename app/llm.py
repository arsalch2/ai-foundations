"""One place that builds the LLM client, so the rest of the code never cares which provider is used.

All three providers speak the OpenAI Chat Completions format, so one SDK covers them.
"""

from openai import AzureOpenAI, OpenAI

from app.config import Settings, get_settings


def build_client(settings: Settings) -> OpenAI:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env")
        return OpenAI(api_key=settings.openai_api_key)

    if settings.llm_provider == "azure":
        if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
            raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
        endpoint = settings.azure_openai_endpoint.rstrip("/")
        if settings.azure_openai_api_version:
            # Older style: dated API version
            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
            )
        # Newer style: OpenAI-compatible v1 endpoint
        return OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=settings.azure_openai_api_key)

    # Ollama ignores the key, but the SDK requires one
    return OpenAI(base_url=settings.ollama_base_url, api_key="ollama")


def get_client() -> OpenAI:
    """FastAPI dependency. Tests replace this with a fake."""
    return build_client(get_settings())
