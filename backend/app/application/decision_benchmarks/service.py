from dataclasses import dataclass

from app.application.datasets.service import DatasetService
from app.application.decisions.registry import (
    DecisionProviderRegistry,
    DecisionProviderRuntimeConfig,
)
from app.application.jobs.executor import JobExecutor, JobSnapshot, TrainingJob
from app.domain.decision.repositories import DecisionRunRepository
from app.domain.decision.runs import DecisionRun, ZeroShotDecisionSpec
from app.domain.decision.specifications import DecisionSpec
from app.domain.exceptions import DecisionBenchmarkError
from app.domain.experiment.spec import TaskType


@dataclass(frozen=True, slots=True)
class DecisionSubmission:
    """Resultado da submissão de um benchmark de decisão contendo a entidade e o job.

    Attributes:
        run (DecisionRun): Entidade persistida com especificações do benchmark.
        job (JobSnapshot): Snapshot do job de execução assíncrona despachado.
    """

    run: DecisionRun
    job: JobSnapshot


class DecisionBenchmarkService:
    """Serviço de aplicação para validação, registro e despacho de benchmarks de decisão.

    Attributes:
        datasets (DatasetService): Serviço de domínio para recuperação de perfis de dados.
        runs (DecisionRunRepository): Repositório para persistência de execuções de decisão.
        providers (DecisionProviderRegistry): Catálogo de provedores registrados.
        provider_config (DecisionProviderRuntimeConfig): Configuração ativa de provedor.
        executor (JobExecutor): Executor de jobs assíncronos.
        database_url (str): String de conexão com o banco de dados.
        dataset_path (str): Diretório base de arquivos de dados brutos.
        artifact_path (str): Diretório base de artefatos gerados.
    """

    def __init__(
        self,
        *,
        datasets: DatasetService,
        runs: DecisionRunRepository,
        providers: DecisionProviderRegistry,
        provider_config: DecisionProviderRuntimeConfig,
        executor: JobExecutor,
        database_url: str,
        dataset_path: str,
        artifact_path: str,
    ) -> None:
        """Inicializa o serviço de benchmark de decisão com suas dependências.

        Args:
            datasets (DatasetService): Serviço de datasets.
            runs (DecisionRunRepository): Repositório de execuções de benchmark.
            providers (DecisionProviderRegistry): Registro de provedores.
            provider_config (DecisionProviderRuntimeConfig): Configuração de provedores.
            executor (JobExecutor): Executor de tarefas assíncronas.
            database_url (str): URL de conexão ao banco.
            dataset_path (str): Caminho para os datasets.
            artifact_path (str): Caminho para gravação de artefatos.
        """
        self.datasets = datasets
        self.runs = runs
        self.providers = providers
        self.provider_config = provider_config
        self.executor = executor
        self.database_url = database_url
        self.dataset_path = dataset_path
        self.artifact_path = artifact_path

    def submit(
        self,
        specification: ZeroShotDecisionSpec,
        *,
        external_processing_acknowledged: bool,
        experiment_id: str | None = None,
    ) -> DecisionSubmission:
        """Valida contrato de colunas, classes e provedor, persistindo a execução e job.

        Args:
            specification (ZeroShotDecisionSpec): Especificação completa do benchmark.
            external_processing_acknowledged (bool): Consentimento do usuário para envio externo.
            experiment_id (str | None): ID de experimento supervisionado associado, opcional.

        Returns:
            DecisionSubmission: Tupla contendo a execução persistida e o snapshot do job.

        Raises:
            DecisionBenchmarkError: Se houver inconformidades de colunas, classes ou provedor.
        """
        self._validate_provider(specification, external_processing_acknowledged)
        dataset = self.datasets.get_dataset(specification.dataset_id)
        profile = self.datasets.get_profile(specification.dataset_id)
        columns = set(profile.column_names)
        unknown = {specification.target, *specification.input_columns} - columns
        if unknown:
            raise DecisionBenchmarkError(f"Unknown dataset columns: {sorted(unknown)}")
        target_profile = next(
            column for column in profile.column_profiles if column.name == specification.target
        )
        target_classes = {item.value for item in target_profile.top_values}
        if target_profile.cardinality < 2:
            raise DecisionBenchmarkError("Benchmark target must contain at least two classes")
        if specification.task == TaskType.BINARY_CLASSIFICATION:
            if target_profile.cardinality != 2:
                raise DecisionBenchmarkError("Binary benchmark target must contain two classes")
            if specification.positive_class not in target_classes:
                raise DecisionBenchmarkError("Positive class is not present in the target")
        else:
            question_classes = set(specification.question.criteria)
            if target_profile.cardinality != len(question_classes) or not target_classes.issubset(
                question_classes
            ):
                raise DecisionBenchmarkError(
                    "Choice criteria must exactly represent the target classes"
                )

        question_spec_hash = DecisionSpec(
            provider=specification.provider,
            model=specification.model,
            locale=specification.question.locale,
            questions={"prediction": specification.question},
        ).question_spec_hash()
        run = self.runs.add(
            DecisionRun.create(
                specification,
                row_count=dataset.row_count,
                question_spec_hash=question_spec_hash,
                experiment_id=experiment_id,
            )
        )
        try:
            job = self.executor.submit(
                TrainingJob(
                    experiment_id=run.id,
                    database_url=self.database_url,
                    dataset_path=self.dataset_path,
                    artifact_path=self.artifact_path,
                )
            )
        except Exception:
            from app.domain.decision.runs import DecisionRunStatus

            self.runs.transition(
                run.id,
                DecisionRunStatus.FAILED,
                error_message="Decision worker could not be started",
            )
            raise
        return DecisionSubmission(run=run, job=job)

    def _validate_provider(
        self,
        specification: ZeroShotDecisionSpec,
        acknowledged: bool,
    ) -> None:
        """Verifica a habilitação do provedor e a confirmação de processamento externo.

        Args:
            specification (ZeroShotDecisionSpec): Especificação do benchmark solicitado.
            acknowledged (bool): Confirmação de processamento externo pelo usuário.

        Raises:
            DecisionBenchmarkError: Se o provedor estiver desabilitado, indisponível ou faltar
                o consentimento de processamento externo.
        """
        if not self.provider_config.enabled:
            raise DecisionBenchmarkError("Decision features are disabled")
        summaries = {item.id: item for item in self.providers.summaries(self.provider_config)}
        aliases = {"typesafe": "typesafe_jev", "openrouter": "openrouter_jev"}
        provider_id = aliases.get(specification.provider, specification.provider)
        summary = summaries.get(provider_id)
        if summary is None or not summary.selected or not summary.available:
            raise DecisionBenchmarkError("Selected Decision Provider is not configured")
        if self.provider_config.external_processing_ack_required and not acknowledged:
            raise DecisionBenchmarkError("External processing acknowledgement is required")
