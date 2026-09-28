from app.application.experiments.service import ExperimentService
from app.application.jobs.executor import (
    TERMINAL_JOB_STATUSES,
    JobExecutor,
    JobSnapshot,
    TrainingJob,
)
from app.domain.exceptions import ExperimentExecutionError, JobAlreadyRunningError
from app.domain.experiment.entities import ExperimentStatus
from app.domain.experiment.spec import ExperimentSpec


class TrainingSubmissionService:
    """Valida e despacha a execução de experimentos para o executor de jobs.

    Attributes:
        experiments (ExperimentService): Serviço de domínio para gerência de experimentos.
        executor (JobExecutor): Executor responsável pelo ciclo de vida dos processos.
        database_url (str): String de conexão com o banco de dados da aplicação.
        dataset_path (str): Diretório base onde os arquivos de datasets estão armazenados.
        artifact_path (str): Diretório base onde os artefatos de treinamento serão salvos.
    """

    def __init__(
        self,
        *,
        experiments: ExperimentService,
        executor: JobExecutor,
        database_url: str,
        dataset_path: str,
        artifact_path: str,
    ) -> None:
        """Inicializa o serviço de submissão com os caminhos e componentes necessários.

        Args:
            experiments (ExperimentService): Serviço de experimentos para consulta e status.
            executor (JobExecutor): Instância do executor de jobs.
            database_url (str): URL de conexão com banco de dados.
            dataset_path (str): Caminho para o diretório de dados.
            artifact_path (str): Caminho para o diretório de artefatos.
        """
        self.experiments = experiments
        self.executor = executor
        self.database_url = database_url
        self.dataset_path = dataset_path
        self.artifact_path = artifact_path

    def submit(self, experiment_id: str) -> JobSnapshot:
        """Valida especificação, atualiza status para QUEUED e enfileira job de treino.

        Args:
            experiment_id (str): Identificador do experimento a ser executado.

        Returns:
            JobSnapshot: Snapshot inicial do job criado.

        Raises:
            JobAlreadyRunningError: Se o experimento já possuir um job ativo em andamento.
            ExperimentExecutionError: Se o processo worker não puder ser inicializado.
        """
        experiment = self.experiments.get(experiment_id)
        self.experiments.validate_specification(ExperimentSpec.model_validate(experiment.spec))
        active = self.executor.find_for_experiment(experiment_id)
        if active is not None and active.status not in TERMINAL_JOB_STATUSES:
            raise JobAlreadyRunningError(f"Experiment {experiment_id} already has an active job")
        self.experiments.update_status(experiment_id, ExperimentStatus.QUEUED)
        try:
            return self.executor.submit(
                TrainingJob(
                    experiment_id=experiment_id,
                    database_url=self.database_url,
                    dataset_path=self.dataset_path,
                    artifact_path=self.artifact_path,
                )
            )
        except Exception as exc:
            self.experiments.update_status(experiment_id, ExperimentStatus.FAILED)
            if isinstance(exc, JobAlreadyRunningError):
                raise
            raise ExperimentExecutionError("Training worker could not be started") from exc
