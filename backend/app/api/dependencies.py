from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import ValidationError

from app.application.advisor.prompts import AdvisorPromptBuilder
from app.application.advisor.service import AdvisorService
from app.application.advisor.validation import AdvisorRecommendationValidator
from app.application.auto.service import AutoWorkflowService
from app.application.confidence_cascade.service import ConfidenceCascadeService
from app.application.datasets.importers import CsvDatasetImporter
from app.application.datasets.profiling import DeterministicDatasetProfiler
from app.application.datasets.service import DatasetService
from app.application.decision_benchmarks.comparison import DecisionComparisonService
from app.application.decision_benchmarks.service import DecisionBenchmarkService
from app.application.decision_benchmarks.worker import (
    cancel_decision_job,
    execute_decision_job,
)
from app.application.decisions.registry import (
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
    build_default_decision_provider_registry,
)
from app.application.experiments.service import ExperimentService
from app.application.experiments.validation import ExperimentSpecValidator
from app.application.explanations.prompts import ExplanationPromptBuilder
from app.application.explanations.service import ExplanationService
from app.application.improvements.capabilities import ImprovementCapabilityCatalog
from app.application.improvements.prompts import ImprovementPromptBuilder
from app.application.improvements.service import ImprovementProposalService
from app.application.improvements.validation import ImprovementProposalValidator
from app.application.jobs.executor import JobExecutor, LocalProcessExecutor
from app.application.jobs.service import TrainingSubmissionService
from app.application.semantic_features.proposer import SemanticFeatureProposer
from app.application.semantic_features.service import SemanticFeatureService
from app.application.semantic_features.worker import (
    cancel_semantic_feature_job,
    execute_semantic_feature_job,
)
from app.core.config.settings import get_settings
from app.infrastructure.database.session import Database
from app.infrastructure.decision.providers import OpenRouterJevProvider, TypeSafeJevProvider
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyCalibrationRunRepository,
    SqlAlchemyCascadeRunRepository,
    SqlAlchemyDatasetRepository,
    SqlAlchemyDecisionCacheRepository,
    SqlAlchemyDecisionRunRepository,
    SqlAlchemyExperimentRepository,
    SqlAlchemyExperimentStrategyRepository,
    SqlAlchemyModelRunRepository,
    SqlAlchemySemanticFeatureSetRepository,
    SqlAlchemyTuningTrialRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.llm.base import LLMProvider, LLMProviderConfig, UnavailableLLMProvider
from app.llm.errors import LLMConfigurationError
from app.llm.registry import build_default_llm_provider_registry
from app.ml.registries.catalog import RegistryCatalog, build_default_registry_catalog


@lru_cache
def get_database() -> Database:
    """Retorna a instância singleton gerenciadora de conexões com o banco de dados.

    Returns:
        Database: Instância configurada com a URL das configurações da aplicação.
    """
    return Database(get_settings().database_url)


@lru_cache
def get_dataset_service() -> DatasetService:
    """Retorna o serviço de domínio e orquestração de datasets configurado.

    Returns:
        DatasetService: Serviço com repositório, storage, importador e profilador.
    """
    settings = get_settings()
    return DatasetService(
        repository=SqlAlchemyDatasetRepository(get_database()),
        storage=LocalFilesystemStorage(settings.dataset_path),
        importer=CsvDatasetImporter(),
        profiler=DeterministicDatasetProfiler(preview_rows=settings.dataset_preview_rows),
        working_directory=f"{settings.dataset_path}/.incoming",
        max_upload_bytes=settings.max_upload_size_mb * 1024 * 1024,
    )


@lru_cache
def get_artifact_storage() -> LocalFilesystemStorage:
    """Retorna o storage local para persistência de artefatos de treinamento e modelos.

    Returns:
        LocalFilesystemStorage: Storage apontando para o diretório de artefatos.
    """
    return LocalFilesystemStorage(get_settings().artifact_path)


@lru_cache
def get_registry_catalog() -> RegistryCatalog:
    """Retorna o catálogo padrão contendo os registros de métricas e adaptadores de ML.

    Returns:
        RegistryCatalog: Instância singleton com catálogo de componentes ML.
    """
    return build_default_registry_catalog()


@lru_cache
def get_decision_provider_registry() -> DecisionProviderRegistry:
    """Retorna o catálogo registrado de provedores de inferência de decisão.

    Returns:
        DecisionProviderRegistry: Instância singleton com provedores configurados.
    """
    return build_default_decision_provider_registry(
        typesafe_factory=TypeSafeJevProvider,
        openrouter_factory=OpenRouterJevProvider,
        typesafe_capabilities=TypeSafeJevProvider.capabilities,
        openrouter_capabilities=OpenRouterJevProvider.capabilities,
    )


def get_decision_provider_runtime_config() -> DecisionProviderRuntimeConfig:
    """Gera a configuração em tempo de execução para os provedores de decisão.

    Returns:
        DecisionProviderRuntimeConfig: Parâmetros lidos dinamicamente das configurações.
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


@lru_cache
def get_decision_run_repository() -> SqlAlchemyDecisionRunRepository:
    """Retorna o repositório SQLAlchemy de execuções de benchmarks de decisão.

    Returns:
        SqlAlchemyDecisionRunRepository: Repositório singleton.
    """
    return SqlAlchemyDecisionRunRepository(get_database())


@lru_cache
def get_calibration_run_repository() -> SqlAlchemyCalibrationRunRepository:
    """Retorna o repositório SQLAlchemy de calibrações de incerteza e temperatura.

    Returns:
        SqlAlchemyCalibrationRunRepository: Repositório singleton.
    """
    return SqlAlchemyCalibrationRunRepository(get_database())


@lru_cache
def get_cascade_run_repository() -> SqlAlchemyCascadeRunRepository:
    """Retorna o repositório SQLAlchemy de execuções de cascata de confiança.

    Returns:
        SqlAlchemyCascadeRunRepository: Repositório singleton.
    """
    return SqlAlchemyCascadeRunRepository(get_database())


def get_confidence_cascade_service() -> ConfidenceCascadeService:
    """Retorna o serviço de simulação e orquestração de cascatas de confiança.

    Returns:
        ConfidenceCascadeService: Serviço configurado com repositórios e storage.
    """
    return ConfidenceCascadeService(
        decision_runs=get_decision_run_repository(),
        cascades=get_cascade_run_repository(),
        storage=get_artifact_storage(),
    )


@lru_cache
def get_decision_cache_repository() -> SqlAlchemyDecisionCacheRepository:
    """Retorna o repositório SQLAlchemy de cache de respostas de decisão.

    Returns:
        SqlAlchemyDecisionCacheRepository: Repositório singleton.
    """
    return SqlAlchemyDecisionCacheRepository(get_database())


@lru_cache
def get_decision_job_executor() -> JobExecutor:
    """Retorna o executor em subprocessos dedicado aos jobs de benchmark de decisão.

    Returns:
        JobExecutor: Instância singleton de executor de processos locais.
    """
    return LocalProcessExecutor(
        worker=execute_decision_job,
        cancellation_handler=cancel_decision_job,
    )


def get_decision_benchmark_service(
    executor: Annotated[JobExecutor, Depends(get_decision_job_executor)],
) -> DecisionBenchmarkService:
    """Retorna o serviço de execução e orquestração de benchmarks de decisão.

    Args:
        executor (JobExecutor): Executor de background jobs injetado via Depends.

    Returns:
        DecisionBenchmarkService: Instância configurada para benchmarks.
    """
    settings = get_settings()
    return DecisionBenchmarkService(
        datasets=get_dataset_service(),
        runs=get_decision_run_repository(),
        providers=get_decision_provider_registry(),
        provider_config=get_decision_provider_runtime_config(),
        executor=executor,
        database_url=settings.database_url,
        dataset_path=settings.dataset_path,
        artifact_path=settings.artifact_path,
    )


def get_decision_comparison_service() -> DecisionComparisonService:
    """Retorna o serviço de comparação entre modelos ML e decisões LLM.

    Returns:
        DecisionComparisonService: Serviço configurado com repositórios e storage.
    """
    return DecisionComparisonService(
        decision_runs=get_decision_run_repository(),
        experiments=SqlAlchemyExperimentRepository(get_database()),
        model_runs=get_model_run_repository(),
        storage=get_artifact_storage(),
    )


@lru_cache
def get_semantic_feature_set_repository() -> SqlAlchemySemanticFeatureSetRepository:
    """Retorna o repositório SQLAlchemy de conjuntos de features semânticas.

    Returns:
        SqlAlchemySemanticFeatureSetRepository: Repositório singleton.
    """
    return SqlAlchemySemanticFeatureSetRepository(get_database())


@lru_cache
def get_semantic_job_executor() -> JobExecutor:
    """Retorna o executor em subprocessos dedicado aos jobs de features semânticas.

    Returns:
        JobExecutor: Instância singleton de executor local.
    """
    return LocalProcessExecutor(
        worker=execute_semantic_feature_job,
        cancellation_handler=cancel_semantic_feature_job,
    )


def get_semantic_feature_proposer() -> SemanticFeatureProposer:
    """Retorna o proponente de features semânticas baseado em LLM.

    Returns:
        SemanticFeatureProposer: Proponente com provedor LLM e templates de prompt.
    """
    settings = get_settings()
    return SemanticFeatureProposer(
        datasets=get_dataset_service(),
        provider=get_llm_provider(),
        prompt_directory=settings.advisor_prompts_path,
        send_sample=settings.llm_send_sample,
        sample_size=settings.llm_sample_size,
        random_seed=settings.default_random_seed,
    )


def get_semantic_feature_service(
    executor: Annotated[JobExecutor, Depends(get_semantic_job_executor)],
) -> SemanticFeatureService:
    """Retorna o serviço de submissão e orquestração de features semânticas.

    Args:
        executor (JobExecutor): Executor de background jobs de features semânticas.

    Returns:
        SemanticFeatureService: Serviço configurado com repositórios e executor.
    """
    settings = get_settings()
    return SemanticFeatureService(
        datasets=get_dataset_service(),
        feature_sets=get_semantic_feature_set_repository(),
        providers=get_decision_provider_registry(),
        provider_config=get_decision_provider_runtime_config(),
        executor=executor,
        database_url=settings.database_url,
        dataset_path=settings.dataset_path,
        artifact_path=settings.artifact_path,
    )


@lru_cache
def get_experiment_spec_validator() -> ExperimentSpecValidator:
    """Retorna o validador de especificações de experimentos de ML.

    Returns:
        ExperimentSpecValidator: Validador com acesso ao catálogo de modelos e métricas.
    """
    return ExperimentSpecValidator(get_registry_catalog())


@lru_cache
def get_experiment_service() -> ExperimentService:
    """Retorna o serviço de gerenciamento do ciclo de vida de experimentos de ML.

    Returns:
        ExperimentService: Serviço com repositórios de experimentos e datasets.
    """
    return ExperimentService(
        experiments=SqlAlchemyExperimentRepository(get_database()),
        datasets=SqlAlchemyDatasetRepository(get_database()),
        validator=get_experiment_spec_validator(),
        load_profile=get_dataset_service().get_profile,
    )


@lru_cache
def get_model_run_repository() -> SqlAlchemyModelRunRepository:
    """Retorna o repositório SQLAlchemy de execuções de modelos treinados.

    Returns:
        SqlAlchemyModelRunRepository: Repositório singleton.
    """
    return SqlAlchemyModelRunRepository(get_database())


@lru_cache
def get_tuning_trial_repository() -> SqlAlchemyTuningTrialRepository:
    """Retorna o repositório SQLAlchemy de trials de otimização de hiperparâmetros.

    Returns:
        SqlAlchemyTuningTrialRepository: Repositório singleton.
    """
    return SqlAlchemyTuningTrialRepository(get_database())


@lru_cache
def get_job_executor() -> JobExecutor:
    """Retorna o executor de processos locais para jobs de treinamento de modelos.

    Returns:
        JobExecutor: Instância singleton de LocalProcessExecutor.
    """
    return LocalProcessExecutor()


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Retorna o cliente do provedor LLM configurado (OpenRouter, Gemini, Ollama).

    Returns:
        LLMProvider: Provedor ativo ou UnavailableLLMProvider em caso de erro.
    """
    settings = get_settings()
    try:
        config = LLMProviderConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            reasoning_model=settings.reasoning_model,
            reasoning=settings.llm_reasoning,
            verbosity=settings.llm_verbosity,
            openrouter_providers=settings.openrouter_providers,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        return build_default_llm_provider_registry().create(config)
    except (ValidationError, LLMConfigurationError):
        return UnavailableLLMProvider(
            settings.llm_provider,
            "The selected LLM provider is not fully configured.",
        )


def get_advisor_service() -> AdvisorService:
    """Retorna o serviço do Advisor para recomendações de modelagem guiadas por LLM.

    Returns:
        AdvisorService: Serviço configurado com datasets, validador e prompts.
    """
    settings = get_settings()
    return AdvisorService(
        datasets=get_dataset_service(),
        provider=get_llm_provider(),
        catalog=get_registry_catalog(),
        validator=AdvisorRecommendationValidator(get_experiment_spec_validator()),
        prompts=AdvisorPromptBuilder(settings.advisor_prompts_path),
        send_sample=settings.llm_send_sample,
        sample_size=settings.llm_sample_size,
        random_seed=settings.default_random_seed,
    )


def get_explanation_service() -> ExplanationService:
    """Retorna o serviço de geração de explicações textuais e narrativas por LLM.

    Returns:
        ExplanationService: Serviço com provedor LLM e templates de prompt.
    """
    settings = get_settings()
    return ExplanationService(
        provider=get_llm_provider(),
        prompts=ExplanationPromptBuilder(settings.advisor_prompts_path),
        random_seed=settings.default_random_seed,
    )


def get_improvement_proposal_service() -> ImprovementProposalService:
    """Retorna o serviço de propostas automáticas de melhoria iterativa de modelos.

    Returns:
        ImprovementProposalService: Serviço com validação de estratégias e prompts.
    """
    settings = get_settings()
    capabilities = ImprovementCapabilityCatalog(get_registry_catalog())
    return ImprovementProposalService(
        provider=get_llm_provider(),
        prompts=ImprovementPromptBuilder(settings.advisor_prompts_path),
        experiments=get_experiment_service(),
        model_runs=get_model_run_repository(),
        capabilities=capabilities,
        validator=ImprovementProposalValidator(
            capabilities=capabilities,
            spec_validator=get_experiment_spec_validator(),
        ),
        persistence=get_experiment_strategy_repository(),
        random_seed=settings.default_random_seed,
    )


@lru_cache
def get_experiment_strategy_repository() -> SqlAlchemyExperimentStrategyRepository:
    """Retorna o repositório SQLAlchemy de estratégias e propostas de melhoria.

    Returns:
        SqlAlchemyExperimentStrategyRepository: Repositório singleton.
    """
    return SqlAlchemyExperimentStrategyRepository(get_database())


def get_training_submission_service(
    experiments: Annotated[ExperimentService, Depends(get_experiment_service)],
    executor: Annotated[JobExecutor, Depends(get_job_executor)],
) -> TrainingSubmissionService:
    """Retorna o serviço de submissão e orquestração de jobs de treinamento.

    Args:
        experiments (ExperimentService): Serviço de experimentos injetado via Depends.
        executor (JobExecutor): Executor de background jobs injetado via Depends.

    Returns:
        TrainingSubmissionService: Serviço configurado com storage e banco.
    """
    settings = get_settings()
    return TrainingSubmissionService(
        experiments=experiments,
        executor=executor,
        database_url=settings.database_url,
        dataset_path=settings.dataset_path,
        artifact_path=settings.artifact_path,
    )


def get_auto_workflow_service(
    advisor: Annotated[AdvisorService, Depends(get_advisor_service)],
    experiments: Annotated[ExperimentService, Depends(get_experiment_service)],
    strategies: Annotated[
        SqlAlchemyExperimentStrategyRepository,
        Depends(get_experiment_strategy_repository),
    ],
    training: Annotated[
        TrainingSubmissionService,
        Depends(get_training_submission_service),
    ],
) -> AutoWorkflowService:
    """Retorna o serviço do fluxo ponta a ponta autônomo (recomendação + treino + refinamento).

    Args:
        advisor (AdvisorService): Serviço do Advisor injetado.
        experiments (ExperimentService): Serviço de experimentos injetado.
        strategies (SqlAlchemyExperimentStrategyRepository): Repositório de estratégias injetado.
        training (TrainingSubmissionService): Serviço de treinamento injetado.

    Returns:
        AutoWorkflowService: Serviço orquestrador do modo autônomo.
    """
    return AutoWorkflowService(
        advisor=advisor,
        experiments=experiments,
        strategies=strategies,
        training=training,
    )
