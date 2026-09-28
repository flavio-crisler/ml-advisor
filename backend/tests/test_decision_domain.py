import asyncio
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from app.application.decisions.errors import DecisionProviderConfigurationError
from app.application.decisions.registry import (
    DecisionProviderRegistration,
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
    build_default_decision_provider_registry,
)
from app.domain.decision import (
    ChoiceDecisionAnswer,
    ChoiceQuestionSpec,
    DecisionProviderCapabilities,
    DecisionResponse,
    DecisionSpec,
    DecisionState,
    DecisionUsage,
    NoulDecisionAnswer,
    NoulQuestionSpec,
    ProviderHealth,
    ScoreDecisionAnswer,
    ScoreQuestionSpec,
)


class MockDecisionProvider:
    id = "mock"
    capabilities = DecisionProviderCapabilities(
        supports_noul=True,
        supports_choice=True,
        supports_score=True,
        is_external=False,
    )

    async def evaluate(
        self,
        state: DecisionState,
        questions: dict[str, Any],
    ) -> DecisionResponse:
        del state, questions
        return DecisionResponse(
            provider_id=self.id,
            requested_model="mock-v1",
            resolved_model="mock-v1.0.0",
            answers={
                "intent": NoulDecisionAnswer(
                    probability_yes=0.8,
                    confidence=0.6,
                )
            },
            usage=DecisionUsage(request_count=1),
        )

    def get_capabilities(self) -> DecisionProviderCapabilities:
        return self.capabilities

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(status="available", latency_ms=1)


def test_question_specs_validate_supported_types() -> None:
    assert NoulQuestionSpec(instructions="Is this urgent?").type == "noul"
    assert ChoiceQuestionSpec(
        instructions="Select the topic.",
        criteria={"billing": "Payments", "technical": "Bugs"},
    ).criteria == {"billing": "Payments", "technical": "Bugs"}
    assert ScoreQuestionSpec(
        instructions="Rate frustration.",
        criteria=["Calm", "Frustrated", "Very frustrated"],
    ).criteria == ["Calm", "Frustrated", "Very frustrated"]


def test_choice_question_normalizes_common_llm_option_metadata() -> None:
    question = ChoiceQuestionSpec.model_validate(
        {
            "instructions": "Classify sentiment.",
            "criteria": {
                "options": "Positive, Neutral, Negative",
                "multi_select": "false",
            },
        }
    )

    assert question.criteria == {
        "Positive": None,
        "Neutral": None,
        "Negative": None,
    }


@pytest.mark.parametrize(
    ("question_type", "payload"),
    [
        (ChoiceQuestionSpec, {"instructions": "Topic", "criteria": {"only-one": None}}),
        (ChoiceQuestionSpec, {"instructions": "Topic", "criteria": {"": "blank"}}),
        (
            ChoiceQuestionSpec,
            {
                "instructions": "Topic",
                "criteria": {"options": "only-one", "multi_select": "false"},
            },
        ),
        (ScoreQuestionSpec, {"instructions": "Score", "criteria": ["only-one"]}),
        (ScoreQuestionSpec, {"instructions": "Score", "criteria": ["low", " "]}),
        (NoulQuestionSpec, {"instructions": "   "}),
    ],
)
def test_invalid_questions_are_rejected(
    question_type: type[NoulQuestionSpec | ChoiceQuestionSpec | ScoreQuestionSpec],
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        question_type.model_validate(payload)


def test_decision_spec_is_versioned_and_hashes_questions_canonically() -> None:
    questions = {
        "topic": ChoiceQuestionSpec(
            instructions="Select the topic.",
            criteria={"billing": "Payments", "technical": "Bugs"},
        ),
        "intent": NoulQuestionSpec(instructions="Is cancellation intended?"),
    }
    first = DecisionSpec(provider="mock", model="mock-v1", questions=questions)
    second = DecisionSpec(
        provider="mock",
        model="another-model",
        questions=dict(reversed(list(questions.items()))),
    )

    assert first.spec_version == "1.0"
    assert first.question_spec_hash() == second.question_spec_hash()


def test_normalized_answers_enforce_probability_invariants() -> None:
    assert (
        ChoiceDecisionAnswer(
            selected_choice="billing",
            choice_probabilities={"billing": 0.7, "technical": 0.3},
            confidence=0.4,
        ).selected_choice
        == "billing"
    )
    assert (
        ScoreDecisionAnswer(
            score=0.75,
            score_probabilities={0: 0.25, 1: 0.75},
            legend={0: "Calm", 1: "Frustrated"},
            confidence=0.6,
        ).score
        == 0.75
    )

    with pytest.raises(ValidationError, match="sum to one"):
        ChoiceDecisionAnswer(
            selected_choice="billing",
            choice_probabilities={"billing": 0.7, "technical": 0.7},
            confidence=0.4,
        )


def test_mock_provider_uses_contract_without_llm_dependency() -> None:
    provider = MockDecisionProvider()
    response = asyncio.run(
        provider.evaluate(
            DecisionState(row_id="row-1", values={"message": "Please cancel"}),
            {"intent": NoulQuestionSpec(instructions="Is cancellation intended?")},
        )
    )

    assert response.provider_id == "mock"
    assert response.resolved_model == "mock-v1.0.0"
    assert provider.get_capabilities().is_external is False
    assert asyncio.run(provider.health_check()).status == "available"


def test_registry_is_extensible_and_never_serializes_secrets() -> None:
    registry = DecisionProviderRegistry()
    registration = DecisionProviderRegistration(
        provider_id="mock",
        display_name="Mock provider",
        capabilities=MockDecisionProvider.capabilities,
        factory=lambda _: MockDecisionProvider(),
    )
    registry.register(registration)
    config = DecisionProviderRuntimeConfig(
        enabled=True,
        provider_id="mock",
        model="mock-v1",
        api_key=SecretStr("never-expose-this"),
    )

    assert registry.create("mock", config).id == "mock"
    assert registry.summaries(config)[0].available is True
    assert "never-expose-this" not in repr(config)
    assert "never-expose-this" not in registry.summaries(config)[0].model_dump_json()

    with pytest.raises(DecisionProviderConfigurationError, match="already registered"):
        registry.register(registration)


def test_default_registry_exposes_provider_neutral_metadata_while_disabled() -> None:
    summaries = build_default_decision_provider_registry().summaries(
        DecisionProviderRuntimeConfig()
    )

    assert [summary.id for summary in summaries] == ["typesafe_jev", "openrouter_jev"]
    assert all(summary.is_external for summary in summaries)
    assert all(summary.health_status == "disabled" for summary in summaries)
    assert all(summary.feature_enabled is False for summary in summaries)
