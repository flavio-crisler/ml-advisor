from dataclasses import dataclass

from app.application.datasets.service import DatasetService
from app.application.decisions.registry import (
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
)
from app.application.jobs.executor import JobExecutor, JobSnapshot, TrainingJob
from app.application.semantic_features.columns import semantic_column_options
from app.domain.decision.repositories import SemanticFeatureSetRepository
from app.domain.decision.semantic_features import (
    SemanticDiscoveryMode,
    SemanticFeatureSet,
    SemanticQuestionProposal,
)
from app.domain.exceptions import DecisionBenchmarkError


@dataclass(frozen=True, slots=True)
class SemanticFeatureSubmission:
    """Resultado da submissão assíncrona de um conjunto de features semânticas.

    Attributes:
        feature_set (SemanticFeatureSet): Entidade persistida do conjunto de features semânticas.
        job (JobSnapshot): Snapshot com o estado inicial do job em segundo plano.
    """

    feature_set: SemanticFeatureSet
    job: JobSnapshot


class SemanticFeatureService:
    """Orquestra a criação, validação e submissão de pipelines de features semânticas.

    Attributes:
        datasets (DatasetService): Serviço para acesso a datasets e perfis estruturais.
        feature_sets (SemanticFeatureSetRepository): Repositório para persistência dos conjuntos.
        providers (DecisionProviderRegistry): Catálogo de provedores de inferência LLM.
        provider_config (DecisionProviderRuntimeConfig): Configuração de execução dos provedores.
        executor (JobExecutor): Executor de background jobs.
        database_url (str): String de conexão com o banco de dados.
        dataset_path (str): Caminho local para armazenamento de arquivos de dados brutos.
        artifact_path (str): Caminho local para persistência de artefatos de modelos e resultados.
    """

    def __init__(
        self,
        *,
        datasets: DatasetService,
        feature_sets: SemanticFeatureSetRepository,
        providers: DecisionProviderRegistry,
        provider_config: DecisionProviderRuntimeConfig,
        executor: JobExecutor,
        database_url: str,
        dataset_path: str,
        artifact_path: str,
    ) -> None:
        """Inicializa o serviço de features semânticas.

        Args:
            datasets (DatasetService): Serviço de datasets.
            feature_sets (SemanticFeatureSetRepository): Repositório de features semânticas.
            providers (DecisionProviderRegistry): Catálogo de provedores de decisão.
            provider_config (DecisionProviderRuntimeConfig): Configurações dos provedores.
            executor (JobExecutor): Executor assíncrono de jobs.
            database_url (str): URL de conexão do banco de dados.
            dataset_path (str): Diretório base de arquivos de dataset.
            artifact_path (str): Diretório base de artefatos de treinamento.
        """
        self.datasets = datasets
        self.feature_sets = feature_sets
        self.providers = providers
        self.provider_config = provider_config
        self.executor = executor
        self.database_url = database_url
        self.dataset_path = dataset_path
        self.artifact_path = artifact_path

    def submit(
        self,
        *,
        dataset_id: str,
        name: str,
        discovery_mode: SemanticDiscoveryMode,
        proposal: SemanticQuestionProposal,
        provider_id: str,
        model: str,
        external_processing_acknowledged: bool,
        source_experiment_id: str | None = None,
    ) -> SemanticFeatureSubmission:
        """Valida, registra e enfileira a execução de extração de features semânticas.

        Args:
            dataset_id (str): Identificador do dataset de origem.
            name (str): Nome amigável atribuído à execução.
            discovery_mode (SemanticDiscoveryMode): Modo de descoberta semântica utilizado.
            proposal (SemanticQuestionProposal): Proposta com as perguntas e colunas envolvidas.
            provider_id (str): Identificador do provedor de decisão (ex: typesafe, openrouter).
            model (str): Nome do modelo a ser utilizado pelo provedor.
            external_processing_acknowledged (bool): Consentimento para envio de dados
                a provedores externos.
            source_experiment_id (str | None): ID opcional do experimento para cálculo
                de impacto preditivo.

        Returns:
            SemanticFeatureSubmission: Objeto contendo o conjunto criado e o snapshot do job.

        Raises:
            DecisionBenchmarkError: Se o modo de descoberta, colunas de entrada ou configurações
                do provedor forem inválidos.
        """
        if discovery_mode != SemanticDiscoveryMode.TARGET_AGNOSTIC:
            raise DecisionBenchmarkError(
                "Target-aware discovery requires an explicit development split"
            )
        dataset = self.datasets.get_dataset(dataset_id)
        profile = self.datasets.get_profile(dataset_id)
        self._validate_provider(provider_id, model, external_processing_acknowledged)
        columns = set(profile.column_names)
        referenced = {
            column for question in proposal.questions for column in question.input_columns
        }
        unknown = referenced - columns
        if unknown:
            raise DecisionBenchmarkError(f"Unknown semantic input columns: {sorted(unknown)}")
        options = semantic_column_options(profile)
        if not referenced.issubset(set(options.recommended + options.optional)):
            raise DecisionBenchmarkError("All semantic inputs must be eligible text columns")
        feature_set = self.feature_sets.add(
            SemanticFeatureSet.create(
                dataset_id=dataset_id,
                source_experiment_id=source_experiment_id,
                name=name.strip(),
                discovery_mode=discovery_mode,
                questions=proposal.questions,
                provider_id=provider_id,
                requested_model=model,
                row_count=dataset.row_count,
            )
        )
        try:
            job = self.executor.submit(
                TrainingJob(
                    experiment_id=feature_set.id,
                    database_url=self.database_url,
                    dataset_path=self.dataset_path,
                    artifact_path=self.artifact_path,
                )
            )
        except Exception:
            from app.domain.decision.runs import DecisionRunStatus

            self.feature_sets.transition(
                feature_set.id,
                DecisionRunStatus.FAILED,
                error_message="Semantic feature worker could not be started",
            )
            raise
        return SemanticFeatureSubmission(feature_set=feature_set, job=job)

    def _validate_provider(self, provider_id: str, model: str, acknowledged: bool) -> None:
        """Valida se o provedor de decisão está ativo, configurado e com termo aceito.

        Args:
            provider_id (str): Identificador do provedor.
            model (str): Nome do modelo solicitado.
            acknowledged (bool): Se o termo de processamento externo foi aceito.

        Raises:
            DecisionBenchmarkError: Se o provedor estiver desabilitado, não configurado
                ou faltar consentimento de processamento externo.
        """
        if not self.provider_config.enabled:
            raise DecisionBenchmarkError("Decision features are disabled")
        aliases = {"typesafe": "typesafe_jev", "openrouter": "openrouter_jev"}
        canonical = aliases.get(provider_id, provider_id)
        summaries = {item.id: item for item in self.providers.summaries(self.provider_config)}
        summary = summaries.get(canonical)
        if (
            summary is None
            or not summary.selected
            or not summary.available
            or model != self.provider_config.model
        ):
            raise DecisionBenchmarkError("Selected Decision Provider is not configured")
        if self.provider_config.external_processing_ack_required and not acknowledged:
            raise DecisionBenchmarkError("External processing acknowledgement is required")
