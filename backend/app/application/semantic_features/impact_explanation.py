from typing import Any

from app.domain.decision.semantic_features import SemanticFeatureSet
from app.domain.experiment.entities import Experiment
from app.ml.registries.catalog import RegistryCatalog


def build_semantic_impact_context(
    feature_set: SemanticFeatureSet,
    experiment: Experiment,
    catalog: RegistryCatalog,
) -> dict[str, Any]:
    """Constrói o contexto agregado de impacto semântico para geração de explicações por LLM.

    Seleciona apenas agregados comparativos e configurações do experimento,
    garantindo que linhas brutas do dataset nunca sejam enviadas ao modelo.

    Args:
        feature_set (SemanticFeatureSet): Conjunto de features semânticas avaliadas.
        experiment (Experiment): Experimento de referência associado.
        catalog (RegistryCatalog): Catálogo de métricas para consulta de direções.

    Returns:
        dict[str, Any]: Estrutura com dados comparativos de modelos e limitações.
    """
    impact = feature_set.impact or {}
    spec = experiment.spec
    models = []
    for model in impact.get("models", []):
        metric = model.get("primary_metric")
        direction = (
            catalog.metrics.get(metric).direction if catalog.metrics.contains(metric) else None
        )
        baseline = model.get("baseline_metrics", {}).get(metric)
        augmented = model.get("augmented_metrics", {}).get(metric)
        models.append(
            {
                "model_id": model.get("model_id"),
                "primary_metric": metric,
                "metric_direction": direction,
                "baseline_value": baseline,
                "with_semantic_variables_value": augmented,
                "with_local_tfidf_value": model.get("text_baseline_metrics", {}).get(metric)
                if model.get("text_baseline_metrics")
                else None,
                "delta_semantic_vs_text": model.get("delta_semantic_vs_text"),
                "delta_absolute": model.get("delta_absolute"),
                "delta_percent": model.get("delta_percent"),
                "duration_seconds": model.get("duration_seconds"),
                "baseline_duration_seconds": model.get("baseline_duration_seconds"),
                "text_baseline_duration_seconds": model.get("text_baseline_duration_seconds"),
                "augmented_duration_seconds": model.get("augmented_duration_seconds"),
                "semantic_feature_importance": [
                    {"feature": entry.get("feature"), "importance": entry.get("importance")}
                    for entry in model.get("semantic_feature_importance", [])[:20]
                ],
            }
        )
    return {
        "source_experiment": {
            "name": experiment.name,
            "objective": experiment.objective,
            "task": spec.get("task"),
            "target": spec.get("target"),
            "source_features": spec.get("features", []),
        },
        "semantic_questions": [
            {
                "id": question.id,
                "input_columns": question.input_columns,
                "instructions": question.question.instructions,
            }
            for question in feature_set.questions
        ],
        "row_count": feature_set.row_count,
        "local_text_columns": impact.get("text_columns", []),
        "comparison_contract": {
            key: impact.get("comparison_contract", {}).get(key)
            for key in (
                "same_dataset",
                "same_validation",
                "same_random_seed",
                "same_models",
                "same_parameters",
                "same_tuning_budget",
                "only_difference",
                "arms",
            )
        },
        "models": models,
        "limitations": [
            "No statistical significance or causal effect was established.",
            "Confirm that all input columns were available at prediction time.",
            "SHAP importance does not demonstrate performance improvement.",
            "duration_seconds includes all available evaluation arms and explainability, "
            "not marginal JEV time.",
        ],
    }
