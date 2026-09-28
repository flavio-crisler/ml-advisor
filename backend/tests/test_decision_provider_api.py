from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.dependencies import (
    get_decision_provider_registry,
    get_decision_provider_runtime_config,
)
from app.application.decisions.registry import (
    DecisionProviderRegistration,
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
)
from app.domain.decision import DecisionProviderCapabilities, ProviderHealth
from app.main import app


class HealthyProvider:
    id = "mock"
    capabilities = DecisionProviderCapabilities(
        supports_noul=True,
        supports_choice=True,
        supports_score=True,
        is_external=True,
    )

    async def evaluate(self, state, questions):  # pragma: no cover - endpoint tests health only
        raise AssertionError((state, questions))

    def get_capabilities(self) -> DecisionProviderCapabilities:
        return self.capabilities

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status="available",
            message_key="decisionProviders.status.connectionAvailable",
            latency_ms=2.5,
        )


def test_decision_provider_registry_endpoint_is_safe_and_in_openapi() -> None:
    app.dependency_overrides[get_decision_provider_runtime_config] = lambda: (
        DecisionProviderRuntimeConfig(
            enabled=True,
            provider_id="typesafe_jev",
            model="jev-alias",
            api_key=SecretStr("private-decision-key"),
        )
    )
    try:
        response = TestClient(app).get("/api/v1/decision-providers")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert [provider["id"] for provider in body] == ["typesafe_jev", "openrouter_jev"]
    assert body[0]["selected"] is True
    assert body[0]["configured"] is True
    assert body[0]["available"] is True
    assert body[0]["model"] == "jev-alias"
    assert "private-decision-key" not in response.text
    assert "/api/v1/decision-providers" in app.openapi()["paths"]


def test_connection_endpoint_uses_configured_registry_provider() -> None:
    registry = DecisionProviderRegistry()
    registry.register(
        DecisionProviderRegistration(
            provider_id="mock",
            display_name="Mock",
            capabilities=HealthyProvider.capabilities,
            factory=lambda _: HealthyProvider(),
        )
    )
    app.dependency_overrides[get_decision_provider_registry] = lambda: registry
    app.dependency_overrides[get_decision_provider_runtime_config] = lambda: (
        DecisionProviderRuntimeConfig(
            enabled=True,
            provider_id="mock",
            model="mock-v1",
            api_key=SecretStr("private-decision-key"),
        )
    )
    try:
        response = TestClient(app).post("/api/v1/decision-providers/test")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "available"
    assert response.json()["latency_ms"] == 2.5
    assert "private-decision-key" not in response.text
    assert "/api/v1/decision-providers/test" in app.openapi()["paths"]


def test_connection_endpoint_rejects_disabled_feature() -> None:
    app.dependency_overrides[get_decision_provider_runtime_config] = lambda: (
        DecisionProviderRuntimeConfig(enabled=False)
    )
    try:
        response = TestClient(app).post("/api/v1/decision-providers/test")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "decision_features_disabled"
