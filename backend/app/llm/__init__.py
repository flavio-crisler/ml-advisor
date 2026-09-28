"""Provider-agnostic structured LLM access."""

from app.llm.base import LLMCapabilities, LLMProvider, LLMProviderConfig, LLMRequest
from app.llm.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTransportError,
)
from app.llm.providers import OllamaProvider, OpenRouterProvider, VLLMProvider
from app.llm.registry import LLMProviderRegistry, build_default_llm_provider_registry

__all__ = [
    "LLMCapabilities",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProviderConfig",
    "LLMProviderError",
    "LLMProviderRegistry",
    "LLMRequest",
    "LLMResponseError",
    "LLMTransportError",
    "OllamaProvider",
    "OpenRouterProvider",
    "VLLMProvider",
    "build_default_llm_provider_registry",
]
