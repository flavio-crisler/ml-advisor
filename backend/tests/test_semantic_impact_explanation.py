import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes import semantic_features as routes
from app.application.explanations.prompts import ExplanationPromptBuilder
from app.application.semantic_features.impact_explanation import build_semantic_impact_context
from app.ml.registries.catalog import build_default_registry_catalog


def test_context_uses_only_aggregates_and_supports_historical_impact() -> None:
    feature_set = SimpleNamespace(
        row_count=150,
        questions=[
            SimpleNamespace(
                id="return_intent",
                input_columns=["Order_Status"],
                question=SimpleNamespace(instructions="Is the order returned?"),
            )
        ],
        impact={
            "validation": {"folds": [{"train_indices": [0, 1], "test_indices": [2]}]},
            "raw_rows": [{"secret": "must-not-leak"}],
            "comparison_contract": {
                "same_validation": True,
                "only_difference": "semantic_features",
            },
            "models": [
                {
                    "model_id": "logistic_regression",
                    "primary_metric": "roc_auc",
                    "baseline_metrics": {"roc_auc": 0.93},
                    "augmented_metrics": {"roc_auc": 0.92},
                    "delta_absolute": -0.01,
                    "delta_percent": -1.075,
                    "duration_seconds": 24.26,
                    "semantic_feature_importance": [
                        {"feature": "semantic_return_intent", "importance": 0.1}
                    ],
                }
            ],
        },
    )
    experiment = SimpleNamespace(
        name="Returns",
        objective="Predict returns",
        spec={
            "task": "binary_classification",
            "target": "Return_Flag",
            "features": ["Order_Status"],
            "rows": [{"secret": "must-not-leak"}],
        },
    )
    context = build_semantic_impact_context(
        feature_set, experiment, build_default_registry_catalog()
    )
    assert context["models"][0]["metric_direction"] == "maximize"
    assert context["models"][0]["baseline_value"] == 0.93
    assert context["models"][0]["with_semantic_variables_value"] == 0.92
    assert "must-not-leak" not in json.dumps(context)
    assert "train_indices" not in json.dumps(context)

    prompt = ExplanationPromptBuilder(Path(__file__).parents[1] / "prompts").build(
        scope="semantic_impact", language="pt-BR", context=context, seed=41
    )
    assert "não a latência adicional" in prompt.system_prompt
    assert "must-not-leak" not in prompt.user_prompt


def test_endpoint_rejects_missing_impact_without_calling_llm() -> None:
    repository = SimpleNamespace(
        get=lambda _: SimpleNamespace(status=SimpleNamespace(value="completed"), impact=None)
    )
    calls = []
    service = SimpleNamespace(explain=lambda **kwargs: calls.append(kwargs))
    with pytest.raises(HTTPException) as error:
        routes.explain_semantic_impact(
            "set-1",
            routes.SemanticImpactExplanationRequest(language="pt-BR"),
            repository,
            service,
            build_default_registry_catalog(),
            None,
        )
    assert error.value.status_code == 409
    assert calls == []
