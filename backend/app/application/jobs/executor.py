import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from multiprocessing import get_context
from typing import Any

from app.application.training.engine import ExperimentEngine
from app.core.config.settings import get_settings
from app.core.logging import configure_logging
from app.domain.common import new_id, utc_now
from app.domain.exceptions import (
    JobAlreadyRunningError,
    JobNotFoundError,
    JobStateError,
)
from app.domain.experiment.entities import ExperimentStatus, ModelRunStatus
from app.infrastructure.database.session import Database
from app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyDatasetRepository,
    SqlAlchemyExperimentRepository,
    SqlAlchemyModelRunRepository,
    SqlAlchemyTuningTrialRepository,
)
from app.infrastructure.storage.local import LocalFilesystemStorage
from app.ml.registries.catalog import build_default_registry_catalog

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Representa o ciclo de vida e estado operacional de um job de treinamento."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """Parâmetros e contexto para inicialização de um job isolado de treinamento.

    Attributes:
        experiment_id (str): Identificador do experimento a ser executado.
        database_url (str): URL de conexão com o banco de dados.
        dataset_path (str): Diretório base de arquivos de datasets.
        artifact_path (str): Diretório base para persistência de artefatos.
        job_id (str): Identificador único do job.
    """

    experiment_id: str
    database_url: str
    dataset_path: str
    artifact_path: str
    job_id: str = field(default_factory=new_id)


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Fotografia do estado instantâneo de um job de treinamento.

    Attributes:
        job_id (str): Identificador único do job.
        experiment_id (str): Identificador do experimento associado.
        status (JobStatus): Status atual do ciclo de vida do job.
        created_at (datetime): Data/hora de enfileiramento em UTC.
        started_at (datetime | None): Data/hora de início da execução em UTC.
        finished_at (datetime | None): Data/hora de término em UTC, se finalizado.
        error (str | None): Mensagem de erro em caso de falha.
    """

    job_id: str
    experiment_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class JobEvent:
    """Evento temporal de progresso emitido durante a execução de um job.

    Attributes:
        sequence (int): Número sequencial monotônico do evento no job.
        type (str): Identificador do tipo do evento emitido.
        timestamp (datetime): Timestamp UTC de emissão do evento.
        payload (dict[str, Any]): Dados contextuais do evento.
    """

    sequence: int
    type: str
    timestamp: datetime
    payload: dict[str, Any]


class JobExecutor(ABC):
    """Interface abstrata para controle e execução de jobs de treinamento."""

    @abstractmethod
    def submit(self, job: TrainingJob) -> JobSnapshot:
        """Enfileira e inicializa a execução de um job de treinamento.

        Args:
            job (TrainingJob): Definição e parâmetros do job a ser executado.

        Returns:
            JobSnapshot: Estado inicial do job criado.
        """
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> JobSnapshot:
        """Solicita o cancelamento imediato de um job em andamento.

        Args:
            job_id (str): Identificador do job a ser cancelado.

        Returns:
            JobSnapshot: Estado atualizado do job após cancelamento.
        """
        ...

    @abstractmethod
    def status(self, job_id: str) -> JobSnapshot:
        """Recupera o estado atual de um job existente.

        Args:
            job_id (str): Identificador do job a ser consultado.

        Returns:
            JobSnapshot: Snapshot contendo status, timestamps e erros.
        """
        ...

    @abstractmethod
    def events(self, job_id: str, *, after: int = 0) -> tuple[JobEvent, ...]:
        """Obtém os eventos de progresso registrados para o job após uma sequência.

        Args:
            job_id (str): Identificador do job.
            after (int): Sequência a partir da qual filtrar eventos.

        Returns:
            tuple[JobEvent, ...]: Tupla de eventos ordenados cronologicamente.
        """
        ...

    @abstractmethod
    def find_for_experiment(self, experiment_id: str) -> JobSnapshot | None:
        """Localiza o job mais recente associado a um determinado experimento.

        Args:
            experiment_id (str): Identificador do experimento consultado.

        Returns:
            JobSnapshot | None: Snapshot do job mais recente ou None se inexistente.
        """
        ...


JobWorker = Callable[[TrainingJob, Callable[[dict[str, Any]], None]], dict[str, Any]]


@dataclass(slots=True)
class _JobState:
    """Estado interno mantido pelo executor para um processo de job ativo.

    Attributes:
        job (TrainingJob): Configurações do job em execução.
        snapshot (JobSnapshot): Snapshot com status e timestamps públicos.
        process (Any): Instância do processo multiprocessing associado.
        event_queue (Any): Fila IPC para comunicação de eventos entre processos.
        events (list[JobEvent]): Histórico cumulativo de eventos registrados.
    """

    job: TrainingJob
    snapshot: JobSnapshot
    process: Any
    event_queue: Any
    events: list[JobEvent] = field(default_factory=list)


def execute_training_job(
    job: TrainingJob, emit: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    """Função alvo executada no processo worker para treinar o experimento.

    Args:
        job (TrainingJob): Configuração e caminhos do job de treinamento.
        emit (Callable[[dict[str, Any]], None]): Callback para emissão de eventos.

    Returns:
        dict[str, Any]: Metadados do resultado contendo IDs de execução e leaderboard.

    Raises:
        Exception: Qualquer exceção levantada pelo motor de experimentos.
    """
    database = Database(job.database_url)
    experiments = SqlAlchemyExperimentRepository(database)
    runs = SqlAlchemyModelRunRepository(database)
    try:
        result = ExperimentEngine(
            datasets=SqlAlchemyDatasetRepository(database),
            experiments=experiments,
            runs=runs,
            tuning_trials=SqlAlchemyTuningTrialRepository(database),
            dataset_storage=LocalFilesystemStorage(job.dataset_path),
            artifact_storage=LocalFilesystemStorage(job.artifact_path),
            catalog=build_default_registry_catalog(),
        ).execute(job.experiment_id, on_event=emit)
        return {
            "experiment_id": result.experiment_id,
            "execution_id": result.execution_id,
            "run_ids": [run.id for run in result.runs],
            "leaderboard_path": result.leaderboard_path,
        }
    except Exception:
        experiment = experiments.get(job.experiment_id)
        if experiment is not None and experiment.status not in {
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        }:
            experiments.update_status(job.experiment_id, ExperimentStatus.FAILED)
        raise
    finally:
        database.dispose()


def _worker_entry(job: TrainingJob, event_queue: Any, worker: JobWorker) -> None:
    """Ponto de entrada isolado do processo filho worker com captura de eventos.

    Args:
        job (TrainingJob): Dados da tarefa de treinamento.
        event_queue (Any): Fila multiprocessing para envio de mensagens ao processo pai.
        worker (JobWorker): Função de trabalho a ser executada.
    """
    configure_logging(get_settings().log_level)

    def emit(payload: dict[str, Any]) -> None:
        event_queue.put({"kind": "event", "payload": payload})

    event_queue.put({"kind": "started"})
    try:
        result = worker(job, emit)
        event_queue.put({"kind": "completed", "result": result})
    except BaseException as error:
        event_queue.put(
            {
                "kind": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )


class LocalProcessExecutor(JobExecutor):
    """Executor que roda jobs de treinamento em subprocessos locais com monitoramento.

    Attributes:
        worker (JobWorker): Função que executa o treinamento dentro do subprocesso.
        context (Any): Contexto de multiprocessing utilizado (ex: spawn).
        cancellation_handler (Callable[[TrainingJob], None]): Função para persistir cancelamento.
    """

    def __init__(
        self,
        worker: JobWorker = execute_training_job,
        *,
        start_method: str = "spawn",
        cancellation_handler: Callable[[TrainingJob], None] | None = None,
    ) -> None:
        """Inicializa o executor de processos locais com o método de spawn especificado.

        Args:
            worker (JobWorker): Função executora dos jobs de treino.
            start_method (str): Método de inicialização de processos multiprocessing.
            cancellation_handler (Callable[[TrainingJob], None] | None): Callback para
                cancelamento.
        """
        self.worker = worker
        self.context = get_context(start_method)
        self.cancellation_handler = cancellation_handler or self._persist_cancellation
        self._states: dict[str, _JobState] = {}
        self._lock = threading.RLock()

    def submit(self, job: TrainingJob) -> JobSnapshot:
        """Inicializa um novo processo worker para o job de treinamento.

        Args:
            job (TrainingJob): Parâmetros e conexões do job.

        Returns:
            JobSnapshot: Snapshot inicial com status QUEUED.

        Raises:
            JobAlreadyRunningError: Se o experimento já tiver um job em andamento.
        """
        with self._lock:
            active = self.find_for_experiment(job.experiment_id)
            if active is not None and active.status not in TERMINAL_JOB_STATUSES:
                raise JobAlreadyRunningError(
                    f"Experiment {job.experiment_id} already has an active job"
                )
            event_queue = self.context.Queue()
            process = self.context.Process(
                target=_worker_entry,
                args=(job, event_queue, self.worker),
                name=f"ml-advisor-job-{job.job_id}",
                daemon=True,
            )
            snapshot = JobSnapshot(
                job_id=job.job_id,
                experiment_id=job.experiment_id,
                status=JobStatus.QUEUED,
                created_at=utc_now(),
            )
            state = _JobState(
                job=job,
                snapshot=snapshot,
                process=process,
                event_queue=event_queue,
            )
            self._states[job.job_id] = state
            self._append_event(state, "job.queued", {"experiment_id": job.experiment_id})
            process.start()
            threading.Thread(
                target=self._monitor,
                args=(job.job_id,),
                name=f"ml-advisor-monitor-{job.job_id}",
                daemon=True,
            ).start()
            return snapshot

    def cancel(self, job_id: str) -> JobSnapshot:
        """Interrompe e cancela um job em execução matando o processo e atualizando o banco.

        Args:
            job_id (str): Identificador do job a ser cancelado.

        Returns:
            JobSnapshot: Snapshot com status CANCELLED.

        Raises:
            JobStateError: Se o job já estiver em estado terminal.
        """
        with self._lock:
            state = self._state(job_id)
            if state.snapshot.status in TERMINAL_JOB_STATUSES:
                raise JobStateError(f"Job {job_id} is already {state.snapshot.status.value}")
            state.snapshot = replace(
                state.snapshot,
                status=JobStatus.CANCELLED,
                finished_at=utc_now(),
            )
            self._append_event(
                state,
                "job.cancelled",
                {"experiment_id": state.job.experiment_id},
            )
            process = state.process
            snapshot = state.snapshot
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self.cancellation_handler(state.job)
        return snapshot

    def status(self, job_id: str) -> JobSnapshot:
        """Consulta o estado atual do job pelo identificador.

        Args:
            job_id (str): Identificador do job.

        Returns:
            JobSnapshot: Snapshot do job.
        """
        with self._lock:
            return self._state(job_id).snapshot

    def events(self, job_id: str, *, after: int = 0) -> tuple[JobEvent, ...]:
        """Retorna eventos de progresso registrados após o índice de sequência fornecido.

        Args:
            job_id (str): Identificador do job.
            after (int): Sequência mínima exclusiva para retorno de eventos.

        Returns:
            tuple[JobEvent, ...]: Eventos emitidos após a sequência especificada.
        """
        with self._lock:
            return tuple(event for event in self._state(job_id).events if event.sequence > after)

    def find_for_experiment(self, experiment_id: str) -> JobSnapshot | None:
        """Busca o job mais recente disparado para o experimento fornecido.

        Args:
            experiment_id (str): Identificador do experimento.

        Returns:
            JobSnapshot | None: O job mais recente encontrado ou None.
        """
        with self._lock:
            matches = [
                state.snapshot
                for state in self._states.values()
                if state.job.experiment_id == experiment_id
            ]
            return max(matches, key=lambda item: item.created_at) if matches else None

    def shutdown(self) -> None:
        """Cancela todos os jobs ativos e encerra o executor de forma graciosa."""
        with self._lock:
            active = [
                state.snapshot.job_id
                for state in self._states.values()
                if state.snapshot.status not in TERMINAL_JOB_STATUSES
            ]
        for job_id in active:
            self.cancel(job_id)

    def _monitor(self, job_id: str) -> None:
        """Loop de monitoramento da fila de mensagens IPC executado em thread dedicada.

        Args:
            job_id (str): Identificador do job monitorado.
        """
        state = self._state(job_id)
        saw_terminal_message = False
        while True:
            with self._lock:
                if state.snapshot.status == JobStatus.CANCELLED:
                    return
            try:
                message = state.event_queue.get(timeout=0.1)
            except queue.Empty:
                if not state.process.is_alive():
                    break
                continue
            kind = message.get("kind")
            with self._lock:
                if state.snapshot.status == JobStatus.CANCELLED:
                    return
                if kind == "started":
                    state.snapshot = replace(
                        state.snapshot,
                        status=JobStatus.RUNNING,
                        started_at=utc_now(),
                    )
                    self._append_event(
                        state,
                        "job.started",
                        {"experiment_id": state.job.experiment_id},
                    )
                elif kind == "event":
                    payload = dict(message["payload"])
                    event_type = str(payload.pop("type"))
                    self._append_event(state, event_type, payload)
                elif kind == "completed":
                    state.snapshot = replace(
                        state.snapshot,
                        status=JobStatus.COMPLETED,
                        finished_at=utc_now(),
                    )
                    self._append_event(state, "job.completed", dict(message["result"]))
                    saw_terminal_message = True
                elif kind == "failed":
                    state.snapshot = replace(
                        state.snapshot,
                        status=JobStatus.FAILED,
                        finished_at=utc_now(),
                        error=str(message["error"]),
                    )
                    self._append_event(
                        state,
                        "job.failed",
                        {
                            "experiment_id": state.job.experiment_id,
                            "error": str(message["error"]),
                        },
                    )
                    saw_terminal_message = True
            if saw_terminal_message:
                break
        state.process.join(timeout=1)
        with self._lock:
            if state.snapshot.status not in TERMINAL_JOB_STATUSES:
                error = f"Worker exited unexpectedly with code {state.process.exitcode}"
                state.snapshot = replace(
                    state.snapshot,
                    status=JobStatus.FAILED,
                    finished_at=utc_now(),
                    error=error,
                )
                self._append_event(
                    state,
                    "job.failed",
                    {"experiment_id": state.job.experiment_id, "error": error},
                )

    def _state(self, job_id: str) -> _JobState:
        """Obtém o estado interno de um job ou levanta exceção se não existir.

        Args:
            job_id (str): Identificador do job.

        Returns:
            _JobState: Estado interno completo do job.

        Raises:
            JobNotFoundError: Se o job não for encontrado no registro interno.
        """
        try:
            return self._states[job_id]
        except KeyError as error:
            raise JobNotFoundError(f"Job {job_id} was not found") from error

    @staticmethod
    def _append_event(state: _JobState, event_type: str, payload: dict[str, Any]) -> None:
        """Anexa um novo evento à lista do job e registra log estruturado correspondente.

        Args:
            state (_JobState): Estado interno do job afetado.
            event_type (str): Identificador do evento emitido.
            payload (dict[str, Any]): Carga de dados associada ao evento.
        """
        state.events.append(
            JobEvent(
                sequence=len(state.events) + 1,
                type=event_type,
                timestamp=utc_now(),
                payload=payload,
            )
        )
        context = {
            "event": event_type,
            "job_id": state.job.job_id,
            "experiment_id": state.job.experiment_id,
        }
        for context_field in ("run_id", "model", "progress"):
            if context_field in payload:
                context[context_field] = payload[context_field]
        level = logging.ERROR if event_type == "job.failed" else logging.INFO
        logger.log(level, event_type, extra=context)

    @staticmethod
    def _persist_cancellation(job: TrainingJob) -> None:
        """Atualiza no banco o status de experimento e runs para CANCELLED.

        Args:
            job (TrainingJob): Job cujo cancelamento deve ser persistido.
        """
        database = Database(job.database_url)
        try:
            experiments = SqlAlchemyExperimentRepository(database)
            runs = SqlAlchemyModelRunRepository(database)
            experiment = experiments.get(job.experiment_id)
            if experiment is not None and experiment.status not in {
                ExperimentStatus.COMPLETED,
                ExperimentStatus.FAILED,
                ExperimentStatus.CANCELLED,
            }:
                experiments.update_status(job.experiment_id, ExperimentStatus.CANCELLED)
            for run in runs.list_for_experiment(job.experiment_id):
                if run.status in {ModelRunStatus.QUEUED, ModelRunStatus.RUNNING}:
                    runs.transition(run.id, ModelRunStatus.CANCELLED)
        finally:
            database.dispose()
