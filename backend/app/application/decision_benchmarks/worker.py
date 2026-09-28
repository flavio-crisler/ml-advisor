from typing import Any

from app.application.decision_benchmarks.execution import DecisionExecutionService
from app.application.decisions.registry import (
    DecisionProviderRuntimeConfig,
    build_default_decision_provider_registry,
)
from app.application.jobs.executor import TrainingJob
from app.core.config.settings import get_settings
from app.domain.decision.runs import DecisionRunStatus
from app.infrastructure.database.session import Database
from app.infrastructure.decision.providers import OpenRouterJevProvider, TypeSafeJevProvider
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyCalibrationRunRepository,
    SqlAlchemyDatasetRepository,
    SqlAlchemyDecisionCacheRepository,
    SqlAlchemyDecisionRunRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage


def build_runtime_config() -> DecisionProviderRuntimeConfig:
    """Carrega as configurações de runtime para provedores de decisão a partir dos settings.

    Returns:
        DecisionProviderRuntimeConfig: Instância com chaves, URLs e limites configurados.
    """
    settings = get_settings()
    return DecisionProviderRuntimeConfig(
        enabled=settings.decision_features_enabled,
        provider_id=settings.decision_provider,
        model=settings.decision_model,
        base_url=settings.decision_base_url,
        api_key=settings.decision_api_key,
        timeout_seconds=settings.decision_request_timeout_seconds,
        max_retries=settings.decision_max_retries,
        concurrency=settings.decision_concurrency,
        cache_enabled=settings.decision_cache_enabled,
        external_processing_ack_required=settings.decision_external_processing_ack_required,
    )


def build_provider_registry():
    """Constrói o catálogo de provedores de decisão com as implementações de infraestrutura.

    Returns:
        DecisionProviderRegistry: Catálogo populado com fábricas TypeSafe e OpenRouter.
    """
    return build_default_decision_provider_registry(
        typesafe_factory=TypeSafeJevProvider,
        openrouter_factory=OpenRouterJevProvider,
        typesafe_capabilities=TypeSafeJevProvider.capabilities,
        openrouter_capabilities=OpenRouterJevProvider.capabilities,
    )


def execute_decision_job(job: TrainingJob, emit: Any) -> dict[str, Any]:
    """Ponto de entrada do processo worker para execução de benchmarks de decisão.

    Args:
        job (TrainingJob): Parâmetros do job e caminhos de banco/armazenamento.
        emit (Any): Callback para despacho de eventos de progresso IPC.

    Returns:
        dict[str, Any]: Dicionário contendo caminhos dos artefatos e métricas finais.
    """
    database = Database(job.database_url)
    try:
        return DecisionExecutionService(
            datasets=SqlAlchemyDatasetRepository(database),
            runs=SqlAlchemyDecisionRunRepository(database),
            calibrations=SqlAlchemyCalibrationRunRepository(database),
            cache=SqlAlchemyDecisionCacheRepository(database),
            dataset_storage=LocalFilesystemStorage(job.dataset_path),
            artifact_storage=LocalFilesystemStorage(job.artifact_path),
            providers=build_provider_registry(),
            provider_config=build_runtime_config(),
        ).execute(job.experiment_id, on_event=emit)
    finally:
        database.dispose()


def cancel_decision_job(job: TrainingJob) -> None:
    """Manipulador de cancelamento para atualizar o status do benchmark para CANCELLED.

    Args:
        job (TrainingJob): Job de benchmark de decisão a ser cancelado.
    """
    database = Database(job.database_url)
    try:
        runs = SqlAlchemyDecisionRunRepository(database)
        run = runs.get(job.experiment_id)
        if run is not None and run.status not in {
            DecisionRunStatus.COMPLETED,
            DecisionRunStatus.FAILED,
            DecisionRunStatus.CANCELLED,
        }:
            runs.transition(run.id, DecisionRunStatus.CANCELLED)
    finally:
        database.dispose()
