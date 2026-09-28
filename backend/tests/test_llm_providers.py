from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, SecretStr

from app.core.config.settings import Settings
from app.llm import (
    LLMCapabilities,
    LLMConfigurationError,
    LLMProvider,
    LLMProviderConfig,
    LLMRequest,
    LLMResponseError,
    OllamaProvider,
    OpenRouterProvider,
    VLLMProvider,
    build_default_llm_provider_registry,
)
from app.llm.transport import JsonTransport


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_ids: list[str]
    explanation: str


class FakeTransport(JsonTransport):
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


@pytest.fixture
def llm_request() -> LLMRequest:
    return LLMRequest(
        system_prompt="Return a valid recommendation.",
        user_prompt="Recommend models for a binary target.",
        language="en-US",
        temperature=0.2,
        seed=41,
    )


@pytest.mark.parametrize(
    ("provider_type", "provider_id", "base_url", "response"),
    [
        (
            OllamaProvider,
            "ollama",
            "http://ollama.test",
            {
                "message": {
                    "content": '{"model_ids":["logistic_regression"],"explanation":"Baseline"}'
                }
            },
        ),
        (
            VLLMProvider,
            "vllm",
            "http://vllm.test/v1",
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"model_ids":["logistic_regression"],"explanation":"Baseline"}'
                            )
                        }
                    }
                ]
            },
        ),
        (
            OpenRouterProvider,
            "openrouter",
            "https://openrouter.test/api/v1",
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"model_ids":["logistic_regression"],"explanation":"Baseline"}'
                            )
                        }
                    }
                ]
            },
        ),
    ],
)
def test_providers_share_the_same_structured_contract(
    provider_type: type[LLMProvider],
    provider_id: str,
    base_url: str,
    response: Mapping[str, Any],
    llm_request: LLMRequest,
) -> None:
    transport = FakeTransport(response)
    config = LLMProviderConfig(
        provider=provider_id,
        model="test-model",
        base_url=base_url,
        api_key=SecretStr("test-secret"),
        reasoning="high",
        timeout_seconds=12,
    )

    provider = provider_type(config, transport)  # type: ignore[call-arg]
    result = provider.generate_structured(llm_request, Recommendation)

    assert result == Recommendation(
        model_ids=["logistic_regression"],
        explanation="Baseline",
    )
    assert provider.capabilities.supports_json_schema is True
    assert transport.calls[0]["timeout_seconds"] == 12
    assert transport.calls[0]["payload"]["model"] == "test-model"


def test_ollama_maps_schema_and_sampling_options(llm_request: LLMRequest) -> None:
    transport = FakeTransport(
        {"message": {"content": {"model_ids": ["catboost"], "explanation": "Mixed data"}}}
    )
    provider = OllamaProvider(
        LLMProviderConfig(provider="ollama", model="qwen3:8b"),
        transport,
    )

    provider.generate_structured(llm_request, Recommendation)

    call = transport.calls[0]
    assert call["url"] == "http://ollama:11434/api/chat"
    assert call["payload"]["format"] == Recommendation.model_json_schema()
    assert call["payload"]["options"] == {"temperature": 0.2, "seed": 41}
    assert "Authorization" not in call["headers"]


def test_openai_compatible_providers_map_json_schema_and_auth(
    llm_request: LLMRequest,
) -> None:
    transport = FakeTransport(
        {"choices": [{"message": {"parsed": {"model_ids": ["xgboost"], "explanation": "Strong"}}}]}
    )
    provider = OpenRouterProvider(
        LLMProviderConfig(
            provider="openrouter",
            model="openai/test",
            api_key=SecretStr("private-key"),
            reasoning_model=True,
            reasoning="medium",
            verbosity="high",
            openrouter_providers=("anthropic", "openai"),
        ),
        transport,
    )

    provider.generate_structured(llm_request, Recommendation)

    call = transport.calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer private-key"
    assert call["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "recommendation",
            "strict": True,
            "schema": Recommendation.model_json_schema(),
        },
    }
    assert call["payload"]["reasoning"] == {"effort": "medium"}
    assert call["payload"]["verbosity"] == "high"
    assert "temperature" not in call["payload"]
    assert call["payload"]["provider"] == {
        "require_parameters": True,
        "order": ["anthropic", "openai"],
        "allow_fallbacks": False,
    }


def test_openrouter_non_reasoning_mode_uses_fixed_temperature_and_ignores_reasoning(
    llm_request: LLMRequest,
) -> None:
    transport = FakeTransport(
        {"choices": [{"message": {"content": '{"model_ids":[],"explanation":"Stable"}'}}]}
    )
    provider = OpenRouterProvider(
        LLMProviderConfig(
            provider="openrouter",
            model="google/test",
            api_key=SecretStr("private-key"),
            reasoning_model=False,
            reasoning="high",
            verbosity="max",
        ),
        transport,
    )

    provider.generate_structured(llm_request, Recommendation)

    payload = transport.calls[0]["payload"]
    assert payload["temperature"] == 0.0
    assert "reasoning" not in payload
    assert "verbosity" not in payload
    assert payload["provider"] == {"require_parameters": True}


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        {"choices": [{"message": {"content": "not-json"}}]},
        {"choices": [{"message": {"content": '{"model_ids":[]}'}}]},
    ],
)
def test_invalid_provider_responses_are_mapped_to_safe_error(
    response: Mapping[str, Any],
    llm_request: LLMRequest,
) -> None:
    provider = VLLMProvider(
        LLMProviderConfig(provider="vllm", model="local-model"),
        FakeTransport(response),
    )

    with pytest.raises(LLMResponseError, match="provider returned"):
        provider.generate_structured(llm_request, Recommendation)


def test_openrouter_requires_a_key_without_exposing_it() -> None:
    with pytest.raises(LLMConfigurationError, match="requires LLM_API_KEY"):
        OpenRouterProvider(LLMProviderConfig(provider="openrouter", model="test"))

    config = LLMProviderConfig(
        provider="openrouter",
        model="test",
        api_key=SecretStr("never-print-this"),
    )
    assert "never-print-this" not in repr(config)


def test_registry_swaps_providers_without_changing_the_consumer(
    llm_request: LLMRequest,
) -> None:
    registry = build_default_llm_provider_registry()
    response = {
        "choices": [
            {"message": {"content": '{"model_ids":["random_forest"],"explanation":"Robust"}'}}
        ]
    }

    def consume(provider: LLMProvider) -> Recommendation:
        return provider.generate_structured(llm_request, Recommendation)

    vllm_result = consume(
        registry.create(
            LLMProviderConfig(provider="vllm", model="local-model"),
            FakeTransport(response),
        )
    )
    openrouter_result = consume(
        registry.create(
            LLMProviderConfig(
                provider="openrouter",
                model="hosted-model",
                api_key=SecretStr("secret"),
            ),
            FakeTransport(response),
        )
    )

    assert vllm_result == openrouter_result
    assert registry.ids() == ("ollama", "vllm", "openrouter")


def test_capabilities_declare_locality_and_unsupported_options() -> None:
    assert OllamaProvider.capabilities == LLMCapabilities(
        supports_json_schema=True,
        supports_temperature=True,
        is_local=True,
    )
    assert VLLMProvider.capabilities.is_local is True
    assert OpenRouterProvider.capabilities.supports_reasoning is True
    assert OpenRouterProvider.capabilities.supports_verbosity is True


def test_settings_keep_api_key_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "settings-secret")
    monkeypatch.setenv("REASONING_MODEL", "true")
    monkeypatch.setenv("OPENROUTER_PROVIDER", " anthropic, openai,anthropic ")

    settings = Settings(_env_file=None)

    assert settings.llm_api_key.get_secret_value() == "settings-secret"
    assert settings.reasoning_model is True
    assert settings.openrouter_providers == ("anthropic", "openai")
    assert "settings-secret" not in repr(settings)
