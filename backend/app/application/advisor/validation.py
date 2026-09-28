from pydantic import ValidationError

from app.application.advisor.errors import AdvisorRecommendationError
from app.application.advisor.models import RecommendationSpec
from app.application.datasets.profiling import ColumnType, DatasetProfile
from app.application.experiments.validation import ExperimentSpecValidator
from app.domain.dataset.entities import Dataset
from app.domain.exceptions import ExperimentSpecValidationError, PipelineCompilationError
from app.domain.experiment.spec import (
    ColumnRole,
    ExperimentSpec,
    ModelSpec,
    PreprocessingStepSpec,
    TaskType,
)
from app.ml.preprocessing.compiler import PipelineCompiler

CONTRACT_ERROR_CODES = {
    "features must be unique": "duplicate_features",
    "target cannot also be a feature": "target_in_features",
    "target must have the target column role": "target_role_missing",
    "exactly one target column role is required": "multiple_targets",
    "every feature must have the feature column role": "feature_role_missing",
    "model ids must be unique": "duplicate_models",
    "preprocessing step ids must be unique": "duplicate_preprocessing_steps",
    "at least one model must be enabled": "no_enabled_model",
    "metric ids must be unique": "duplicate_metrics",
    "multiclass precision, recall, and f1 require averaging": "multiclass_averaging_required",
    "multiclass roc_auc requires an explicit strategy": "multiclass_roc_strategy_required",
    "binary threshold and ranking metrics require positive_label": "binary_positive_label_required",
    "enabled tuning requires an optimization metric": "tuning_metric_required",
}
SAFE_CONTRACT_ISSUES = frozenset(
    f"invalid_contract:{code}" for code in CONTRACT_ERROR_CODES.values()
)


def _contract_issue(error: dict) -> str:
    """Mapeia um erro de validação do Pydantic para um código de contrato padronizado.

    Args:
        error: Dicionário contendo os detalhes da falha reportada pelo Pydantic.

    Returns:
        String com o código seguro de erro de contrato.
    """
    if error["type"] == "value_error":
        cause = str(error.get("ctx", {}).get("error", ""))
        if code := CONTRACT_ERROR_CODES.get(cause):
            return f"invalid_contract:{code}"
    location = ".".join(str(part) for part in error["loc"])
    return f"invalid_contract:{location}:{error['type']}"


class AdvisorRecommendationValidator:
    """Validador de recomendações geradas pelo Advisor e conversor para ExperimentSpec.

    Attributes:
        spec_validator (ExperimentSpecValidator): Validador de especificações de experimentos.
    """

    def __init__(self, spec_validator: ExperimentSpecValidator) -> None:
        """Inicializa o validador com a instância de ExperimentSpecValidator.

        Args:
            spec_validator: Validador de regras de negócio de experimentos.
        """
        self.spec_validator = spec_validator

    def to_proposal(
        self,
        recommendation: RecommendationSpec,
        *,
        dataset: Dataset,
        profile: DatasetProfile,
        objective: str,
        name: str | None = None,
    ) -> ExperimentSpec:
        """Valida a recomendação e a converte em uma proposta de ExperimentSpec executável.

        Aplica normalização defensiva para evitar falhas espúrias causadas por alucinações
        ou repetições do modelo LLM (como duplicação de transformadores, inclusão do target
        nas features ou métricas secundárias repetidas).

        Args:
            recommendation: Recomendação estruturada gerada pelo provedor LLM.
            dataset: Entidade do conjunto de dados associado.
            profile: Perfil estatístico detalhado do dataset.
            objective: Objetivo original informado pelo usuário.
            name: Nome opcional customizado para o experimento.

        Returns:
            ExperimentSpec validada e pronta para execução ou edição pelo usuário.

        Raises:
            AdvisorRecommendationError: Se houver qualquer incompatibilidade semântica, de colunas,
                de contrato ou na compilação do pré-processamento.
        """
        clean_features = self._normalize_features(recommendation)
        clean_preprocessing = self._normalize_preprocessing(recommendation.preprocessing)
        clean_models = self._normalize_models(recommendation.models)
        clean_secondary = [
            m
            for m in dict.fromkeys(recommendation.metrics.secondary)
            if m != recommendation.metrics.primary
        ]
        clean_metrics = recommendation.metrics.model_copy(update={"secondary": clean_secondary})

        issues = self._column_issues(
            recommendation, profile, clean_features=clean_features
        )
        if issues:
            raise AdvisorRecommendationError(issues)

        selected_features = set(clean_features)
        column_roles = {
            column: (
                ColumnRole.TARGET
                if column == recommendation.target
                else ColumnRole.FEATURE
                if column in selected_features
                else ColumnRole.ID
                if column in profile.possible_id_columns
                else ColumnRole.IGNORED
            )
            for column in profile.column_names
        }
        proposal_name = (name or f"{dataset.name} - {recommendation.target}").strip()[:200]
        try:
            proposal = ExperimentSpec(
                dataset_id=dataset.id,
                dataset_hash=dataset.content_hash,
                profile_version=profile.profile_version,
                name=proposal_name,
                objective=objective,
                task=recommendation.task,
                target=recommendation.target,
                features=clean_features,
                column_roles=column_roles,
                models=clean_models,
                preprocessing=clean_preprocessing,
                validation=recommendation.validation,
                metrics=clean_metrics,
                tuning=recommendation.tuning,
            )
            self.spec_validator.validate(proposal)
            column_types = {
                column.name: column.inferred_type.value for column in profile.column_profiles
            }
            for model in (model for model in proposal.models if model.enabled):
                capabilities = self.spec_validator.catalog.models.get(model.id).capabilities
                PipelineCompiler().compile(
                    proposal, column_types=column_types, capabilities=capabilities
                )
        except ValidationError as exc:
            contract_issues = [_contract_issue(error) for error in exc.errors()]
            raise AdvisorRecommendationError(contract_issues) from exc
        except ExperimentSpecValidationError as exc:
            raise AdvisorRecommendationError(list(exc.issues)) from exc
        except PipelineCompilationError as exc:
            raise AdvisorRecommendationError([f"preprocessing_incompatible:{exc}"]) from exc
        return proposal

    @staticmethod
    def _normalize_features(recommendation: RecommendationSpec) -> list[str]:
        """Remove o target e deduplica as features preservando a ordem original.

        Args:
            recommendation: Recomendação contendo as features sugeridas e o alvo.

        Returns:
            Lista de features limpas sem duplicações e sem o target.
        """
        return [
            feature
            for feature in dict.fromkeys(recommendation.features)
            if feature != recommendation.target
        ]

    @staticmethod
    def _normalize_preprocessing(
        steps: list[PreprocessingStepSpec],
    ) -> list[PreprocessingStepSpec]:
        """Deduplica etapas de pré-processamento por identificador mantendo a primeira ocorrência.

        Evita que modelos LLM gerem múltiplos passos para o mesmo ID (por exemplo, dois passos
        de 'missing_values' para tipos numéricos e categóricos separados), mantendo a integridade
        do contrato do ExperimentSpec.

        Args:
            steps: Lista de especificações de etapas de pré-processamento.

        Returns:
            Lista de etapas com IDs únicos.
        """
        seen_ids: set[str] = set()
        unique_steps: list[PreprocessingStepSpec] = []
        for step in steps:
            if step.id not in seen_ids:
                seen_ids.add(step.id)
                unique_steps.append(step)
        return unique_steps

    @staticmethod
    def _normalize_models(models: list[ModelSpec]) -> list[ModelSpec]:
        """Deduplica os modelos propostos por identificador mantendo a primeira ocorrência.

        Args:
            models: Lista de especificações de modelos sugeridos.

        Returns:
            Lista com modelos únicos por ID.
        """
        seen_ids: set[str] = set()
        unique_models: list[ModelSpec] = []
        for model in models:
            if model.id not in seen_ids:
                seen_ids.add(model.id)
                unique_models.append(model)
        return unique_models

    @staticmethod
    def _column_issues(
        recommendation: RecommendationSpec,
        profile: DatasetProfile,
        clean_features: list[str],
    ) -> list[str]:
        """Identifica inconsistências em colunas selecionadas frente ao perfil do dataset.

        Valida se o target existe, se as features limpas existem e não são vazias, e se a
        cardinalidade/tipo do alvo é compatível com a tarefa de modelagem.

        Args:
            recommendation: Recomendação contendo target, features e tipo de tarefa.
            profile: Perfil estatístico com as colunas reais do dataset.
            clean_features: Lista de features após remoção do target e deduplicação.

        Returns:
            Lista de mensagens de inconsistências encontradas (vazia se for válido).
        """
        issues: list[str] = []
        columns = set(profile.column_names)
        if recommendation.target not in columns:
            issues.append(f"unknown_target:{recommendation.target}")
        unknown_features = [
            feature for feature in clean_features if feature not in columns
        ]
        issues.extend(f"unknown_feature:{feature}" for feature in unknown_features)
        if not clean_features:
            issues.append(
                "target_in_features"
                if recommendation.target in recommendation.features
                else "no_features_selected"
            )
        if recommendation.target not in columns:
            return issues

        target_profile = next(
            item for item in profile.column_profiles if item.name == recommendation.target
        )
        if (
            recommendation.task == TaskType.BINARY_CLASSIFICATION
            and target_profile.cardinality != 2
        ):
            issues.append("binary_target_requires_two_classes")
        if (
            recommendation.task == TaskType.MULTICLASS_CLASSIFICATION
            and target_profile.cardinality < 3
        ):
            issues.append("multiclass_target_requires_three_classes")
        if (
            recommendation.task == TaskType.REGRESSION
            and target_profile.inferred_type != ColumnType.NUMERIC
        ):
            issues.append("regression_target_must_be_numeric")
        return issues
