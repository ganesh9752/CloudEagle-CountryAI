import os
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    AZURE_FOUNDRY = "azure_foundry"


class Settings:
    def __init__(self) -> None:
        raw_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        try:
            self.llm_provider = LLMProvider(raw_provider)
        except ValueError:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{raw_provider}'. "
                f"Choose from: {[p.value for p in LLMProvider]}"
            )

        self.llm_model: Optional[str] = os.getenv("LLM_MODEL")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self.openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
        self.azure_foundry_api_key: Optional[str] = os.getenv("AZURE_FOUNDRY_API_KEY")
        self.azure_foundry_endpoint: Optional[str] = os.getenv("AZURE_FOUNDRY_ENDPOINT")

        self._validate()

    def _validate(self) -> None:
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic")
        if self.llm_provider == LLMProvider.OPENROUTER and not self.openrouter_api_key:
            raise EnvironmentError("OPENROUTER_API_KEY must be set when LLM_PROVIDER=openrouter")
        if self.llm_provider == LLMProvider.AZURE_FOUNDRY:
            if not self.azure_foundry_api_key:
                raise EnvironmentError("AZURE_FOUNDRY_API_KEY must be set when LLM_PROVIDER=azure_foundry")
            if not self.azure_foundry_endpoint:
                raise EnvironmentError("AZURE_FOUNDRY_ENDPOINT must be set when LLM_PROVIDER=azure_foundry")


settings = Settings()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm():
    """Return a configured LangChain chat model based on env settings."""
    if settings.llm_provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        model = settings.llm_model or "gpt-4o-mini"
        return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0)

    if settings.llm_provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        model = settings.llm_model or "claude-sonnet-4-6"
        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=0)

    if settings.llm_provider == LLMProvider.OPENROUTER:
        from langchain_openai import ChatOpenAI

        model = settings.llm_model or "nvidia/llama-3.1-nemotron-70b-instruct"
        return ChatOpenAI(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
        )

    if settings.llm_provider == LLMProvider.AZURE_FOUNDRY:
        from langchain_openai import ChatOpenAI

        model = settings.llm_model or "claude-3-5-haiku"
        return ChatOpenAI(
            model=model,
            api_key=settings.azure_foundry_api_key,
            base_url=settings.azure_foundry_endpoint,
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
