from typing import Any

from app.application.datasets.profiling import DatasetProfile
from app.application.decision_benchmarks.worker import (
    build_provider_registry,
    build_runtime_config,
)
from app.application.jobs.executor import TrainingJob
from app.application.semantic_features.execution import SemanticFeatureExecutionService
from app.application.semantic_features.impact import SemanticImpactEvaluator
from app.domain.decision.runs import DecisionRunStatus
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyDecisionCacheRepository,
    SqlAlchemyExperimentRepository,
    SqlAlchemySemanticFeatureSetRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.ml.registries.catalog import build_default_registry_catalog


def execute_semantic_feature_job(job: TrainingJob, emit: Any) -> dict[str, Any]:
    """Executa um job de processamento de features semânticas em processo worker dedicado.

    Args:
        job (TrainingJob): Definição do job contendo parâmetros, banco e paths de storage.
        emit (Any): Função callback síncrona para envio de eventos de progresso.

    Returns:
        dict[str, Any]: Metadados do resultado com ID e caminho do arquivo salvo.

    Raises:
        ValueError: Se o conjunto de features ou perfil do dataset não forem localizados.
    """
    database = Database(job.database_url)
    dataset_storage = LocalFilesystemStorage(job.dataset_path)
    feature_sets = SqlAlchemySemanticFeatureSetRepository(database)
    try:
        feature_set = feature_sets.get(job.experiment_id)
        if feature_set is None:
            raise ValueError(f"Semantic feature set {job.experiment_id} was not found")
        dataset = SqlAlchemyDatasetRepository(database).get(feature_set.dataset_id)
        if dataset is None or dataset.profile_path is None:
            raise ValueError("Dataset profile is unavailable")
        profile = DatasetProfile.model_validate_json(dataset_storage.load(dataset.profile_path))
        column_types = {
            column.name: column.inferred_type.value for column in profile.column_profiles
        }
        return SemanticFeatureExecutionService(
            datasets=SqlAlchemyDatasetRepository(database),
            feature_sets=feature_sets,
            cache=SqlAlchemyDecisionCacheRepository(database),
            dataset_storage=dataset_storage,
            artifact_storage=LocalFilesystemStorage(job.artifact_path),
            providers=build_provider_registry(),
            provider_config=build_runtime_config(),
            impact_evaluator=SemanticImpactEvaluator(
                experiments=SqlAlchemyExperimentRepository(database),
                catalog=build_default_registry_catalog(),
                column_types=column_types,
            ),
        ).execute(job.experiment_id, on_event=emit)
    finally:
        database.dispose()


def cancel_semantic_feature_job(job: TrainingJob) -> None:
    """Cancela um job de features semânticas em execução, transicionando seu status.

    Args:
        job (TrainingJob): Definição do job que deve ser cancelado.
    """
    database = Database(job.database_url)
    try:
        feature_sets = SqlAlchemySemanticFeatureSetRepository(database)
        feature_set = feature_sets.get(job.experiment_id)
        if feature_set is not None and feature_set.status not in {
            DecisionRunStatus.COMPLETED,
            DecisionRunStatus.FAILED,
            DecisionRunStatus.CANCELLED,
        }:
            feature_sets.transition(feature_set.id, DecisionRunStatus.CANCELLED)
    finally:
        database.dispose()
