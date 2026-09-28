import asyncio
import logging
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError

import pytest
from pydantic import SecretStr

from app.application.decisions.errors import (
    DecisionProviderConfigurationError,
    DecisionProviderResponseError,
    DecisionProviderTransportError,
)
from app.application.decisions.registry import DecisionProviderRuntimeConfig
from app.domain.decision import (
    ChoiceQuestionSpec,
    DecisionState,
    NoulQuestionSpec,
    ScoreQuestionSpec,
)
from app.infrastructure.decision.http import (
    DecisionHttpClient,
    DecisionJsonTransport,
    RetryPolicy,
    UrllibDecisionJsonTransport,
)
from app.infrastructure.decision.providers import OpenRouterJevProvider, TypeSafeJevProvider


class FakeDecisionTransport(DecisionJsonTransport):
    def __init__(self, responses: list[Mapping[str, Any] | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def request_json(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def runtime_config(provider_id: str) -> DecisionProviderRuntimeConfig:
    return DecisionProviderRuntimeConfig(
        enabled=True,
        provider_id=provider_id,
        model="jev-latest",
        api_key=SecretStr("private-provider-key"),
        timeout_seconds=12,
        max_retries=2,
    )


def questions() -> dict[str, Any]:
    return {
        "urgent": NoulQuestionSpec(
            instructions="Is this urgent?",
            criteria={"true": "Time-sensitive", "false": "Not time-sensitive"},
            locale="en-US",
        ),
        "topic": ChoiceQuestionSpec(
            instructions="Which team should handle this?",
            criteria={"billing": "Payments", "technical": "Bugs"},
            locale="en-US",
        ),
        "frustration": ScoreQuestionSpec(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated"],
            locale="en-US",
        ),
    }


def provider_response(*, openrouter: bool) -> dict[str, Any]:
    response: dict[str, Any] = {
        "model": "typesafe/jev-1.13-20260917" if openrouter else "jev-1.13.0",
        "answers": {
            "urgent": {"type": "noul", "noul": 0.8},
            "topic": {
                "type": "choice",
                "choice": "technical",
                "confidence": 0.22,
                "probabilities": {"billing": 0.39, "technical": 0.61},
            },
            "frustration": {
                "type": "score",
                "score": 0.75,
                "confidence": 0.5,
                "legend": {"0": "Calm", "1": "Frustrated"},
                "probabilities": {"0": 0.25, "1": 0.75},
            },
        },
        "usage": {"input_tokens": 275, "output_tokens": 20},
    }
    if openrouter:
        response.update({"id": "gen-dec-safe", "provider": "TypeSafe"})
        response["usage"]["cost"] = 0.00003
    return response


@pytest.mark.parametrize(
    ("provider_type", "provider_id", "base_url", "openrouter"),
    [
        (TypeSafeJevProvider, "typesafe_jev", "https://api.typesafe.ai", False),
        (OpenRouterJevProvider, "openrouter_jev", "https://openrouter.ai/api", True),
    ],
)
def test_jev_adapters_share_normalized_contract_and_serialize_native_questions(
    provider_type: type[TypeSafeJevProvider | OpenRouterJevProvider],
    provider_id: str,
    base_url: str,
    openrouter: bool,
) -> None:
    transport = FakeDecisionTransport([provider_response(openrouter=openrouter)])
    client = DecisionHttpClient(
        transport=transport,
        retry_policy=RetryPolicy(max_retries=0),
    )
    provider = provider_type(runtime_config(provider_id), client)

    response = asyncio.run(
        provider.evaluate(
            DecisionState(
                row_id="internal-row-id",
                values={"message": "Please fix this now", "plan": "pro"},
            ),
            questions(),
        )
    )

    call = transport.calls[0]
    assert call["url"] == f"{base_url}/v1/systemone"
    assert call["headers"]["Authorization"] == "Bearer private-provider-key"
    assert call["timeout_seconds"] == 12
    assert call["payload"]["model"] == "jev-latest"
    assert call["payload"]["state"] == {
        "message": "Please fix this now",
        "plan": "pro",
    }
    assert "internal-row-id" not in repr(call["payload"])
    assert call["payload"]["questions"] == {
        "urgent": {
            "type": "noul",
            "instructions": "Is this urgent?",
            "criteria": {"true": "Time-sensitive", "false": "Not time-sensitive"},
        },
        "topic": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments", "technical": "Bugs"},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated"],
        },
    }
    assert response.provider_id == provider_id
    assert response.requested_model == "jev-latest"
    assert response.answers["urgent"].confidence == pytest.approx(0.6)
    assert response.answers["topic"].confidence == 0.22
    assert response.answers["frustration"].confidence == 0.5
    assert response.usage.input_tokens == 275
    assert response.usage.estimated_cost == (0.00003 if openrouter else None)
    assert response.request_id == ("gen-dec-safe" if openrouter else None)


def test_http_client_retries_transient_failure_and_honors_retry_after() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    transport = FakeDecisionTransport(
        [
            DecisionProviderTransportError(
                "rate limited",
                retryable=True,
                status_code=429,
                retry_after_seconds=1.25,
            ),
            {"models": []},
        ]
    )
    client = DecisionHttpClient(
        transport=transport,
        retry_policy=RetryPolicy(max_retries=2, sleep=record_sleep),
    )

    result = asyncio.run(client.get_json("https://provider.test/v1/models", {}, 5))

    assert result == {"models": []}
    assert delays == [1.25]
    assert len(transport.calls) == 2


def test_provider_http_520_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_with_proxy_error(*args: Any, **kwargs: Any) -> Any:
        raise HTTPError("https://provider.test/v1/systemone", 520, "Unknown Error", {}, None)

    monkeypatch.setattr("app.infrastructure.decision.http.urlopen", fail_with_proxy_error)
    with pytest.raises(DecisionProviderTransportError) as failure:
        UrllibDecisionJsonTransport._request_json(
            "POST", "https://provider.test/v1/systemone", {}, {"model": "test"}, 5
        )
    assert failure.value.status_code == 520
    assert failure.value.retryable is True


def test_non_retryable_failure_is_not_repeated() -> None:
    transport = FakeDecisionTransport(
        [
            DecisionProviderTransportError(
                "invalid request",
                retryable=False,
                status_code=422,
            )
        ]
    )
    client = DecisionHttpClient(
        transport=transport,
        retry_policy=RetryPolicy(max_retries=3),
    )

    with pytest.raises(DecisionProviderTransportError):
        asyncio.run(client.get_json("https://provider.test/v1/models", {}, 5))
    assert len(transport.calls) == 1


def test_rounded_provider_probabilities_are_normalized() -> None:
    response = provider_response(openrouter=True)
    response["answers"]["topic"]["probabilities"] = {
        "billing": 0.01,
        "technical": 0.98,
    }
    response["answers"]["frustration"].update(
        {
            "score": 0.75,
            "probabilities": {"0": 0.25, "1": 0.74},
        }
    )
    provider = OpenRouterJevProvider(
        runtime_config("openrouter_jev"),
        DecisionHttpClient(
            transport=FakeDecisionTransport([response]),
            retry_policy=RetryPolicy(max_retries=0),
        ),
    )

    result = asyncio.run(
        provider.evaluate(
            DecisionState(row_id="row", values={"message": "test"}),
            questions(),
        )
    )

    assert sum(result.answers["topic"].choice_probabilities.values()) == pytest.approx(1)
    assert sum(result.answers["frustration"].score_probabilities.values()) == pytest.approx(1)
    assert result.answers["frustration"].score == pytest.approx(0.74 / 0.99)


def test_retry_logging_omits_credentials_and_row_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = FakeDecisionTransport(
        [
            DecisionProviderTransportError("temporary", retryable=True),
            provider_response(openrouter=False),
        ]
    )

    async def skip_sleep(_: float) -> None:
        return None

    provider = TypeSafeJevProvider(
        runtime_config("typesafe_jev"),
        DecisionHttpClient(
            transport=transport,
            retry_policy=RetryPolicy(max_retries=1, sleep=skip_sleep),
        ),
    )
    caplog.set_level(logging.WARNING)

    asyncio.run(
        provider.evaluate(
            DecisionState(row_id="private-row", values={"message": "private-row-content"}),
            questions(),
        )
    )

    assert "private-provider-key" not in caplog.text
    assert "private-row-content" not in caplog.text
    assert any(getattr(record, "event", None) == "decision.retry" for record in caplog.records)


@pytest.mark.parametrize(
    "response",
    [
        {"model": "jev-1.13.0", "answers": {}, "usage": {}},
        {
            "model": "jev-1.13.0",
            "answers": {
                "urgent": {"type": "choice", "choice": "yes"},
                "topic": {},
                "frustration": {},
            },
            "usage": {},
        },
        {
            "model": "jev-1.13.0",
            "answers": {
                "urgent": {"type": "noul", "noul": 0.8},
                "topic": {
                    "type": "choice",
                    "choice": "technical",
                    "confidence": 0.5,
                    "probabilities": {"billing": 0.8, "technical": 0.8},
                },
                "frustration": {
                    "type": "score",
                    "score": 0,
                    "confidence": 1,
                    "probabilities": {"0": 1},
                },
            },
            "usage": {},
        },
    ],
)
def test_malformed_provider_response_is_rejected(response: Mapping[str, Any]) -> None:
    provider = TypeSafeJevProvider(
        runtime_config("typesafe_jev"),
        DecisionHttpClient(
            transport=FakeDecisionTransport([response]),
            retry_policy=RetryPolicy(max_retries=0),
        ),
    )

    with pytest.raises(DecisionProviderResponseError, match="Decision Provider returned"):
        asyncio.run(
            provider.evaluate(
                DecisionState(row_id="row", values={"message": "test"}),
                questions(),
            )
        )


@pytest.mark.parametrize("provider_type", [TypeSafeJevProvider, OpenRouterJevProvider])
def test_provider_requires_model_and_api_key_without_exposing_secrets(
    provider_type: type[TypeSafeJevProvider | OpenRouterJevProvider],
) -> None:
    provider_id = provider_type.id
    with pytest.raises(DecisionProviderConfigurationError, match="model"):
        provider_type(DecisionProviderRuntimeConfig(enabled=True, provider_id=provider_id))
    config = DecisionProviderRuntimeConfig(
        enabled=True,
        provider_id=provider_id,
        model="jev-latest",
        api_key=SecretStr("never-print-provider-key"),
    )
    assert "never-print-provider-key" not in repr(config)


@pytest.mark.parametrize(
    ("provider_type", "provider_id", "models_response"),
    [
        (TypeSafeJevProvider, "typesafe_jev", {"models": []}),
        (OpenRouterJevProvider, "openrouter_jev", {"data": []}),
    ],
)
def test_provider_health_check_uses_authenticated_model_listing(
    provider_type: type[TypeSafeJevProvider | OpenRouterJevProvider],
    provider_id: str,
    models_response: Mapping[str, Any],
) -> None:
    transport = FakeDecisionTransport([models_response])
    provider = provider_type(
        runtime_config(provider_id),
        DecisionHttpClient(
            transport=transport,
            retry_policy=RetryPolicy(max_retries=0),
        ),
    )

    health = asyncio.run(provider.health_check())

    assert health.status == "available"
    assert transport.calls[0]["method"] == "GET"
    assert transport.calls[0]["url"].endswith("/v1/models")
