import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.domain.common import canonical_json, ensure_utc, new_id, utc_now
from app.domain.dataset.entities import Dataset
from app.domain.decision.cascade import CascadeRun
from app.domain.decision.runs import (
    CalibrationRun,
    DecisionRun,
    DecisionRunStatus,
)
from app.domain.decision.semantic_features import (
    SemanticDiscoveryMode,
    SemanticFeatureSet,
)
from app.domain.exceptions import (
    EntityNotFoundError,
    ImmutableExperimentError,
    ImmutableModelRunError,
)
from app.domain.experiment.entities import (
    Experiment,
    ExperimentMode,
    ExperimentStatus,
    ExperimentStrategy,
    ModelRun,
    ModelRunStatus,
    TuningTrial,
    TuningTrialStatus,
)
from app.infrastructure.database.models import (
    CalibrationRunRecord,
    CascadeRunRecord,
    DatasetRecord,
    DecisionCacheRecord,
    DecisionRunRecord,
    ExperimentRecord,
    ExperimentRunRecord,
    ExperimentStrategyRecord,
    SemanticFeatureSetRecord,
    SettingRecord,
    TuningTrialRecord,
)
from app.infrastructure.database.session import Database


def _dataset(record: DatasetRecord) -> Dataset:
    """Converte um registro SQLAlchemy DatasetRecord na entidade de domínio Dataset.

    Args:
        record (DatasetRecord): Registro de banco de dados do dataset.

    Returns:
        Dataset: Instância da entidade de domínio correspondente.
    """
    return Dataset(
        id=record.id,
        name=record.name,
        original_filename=record.original_filename,
        content_hash=record.content_hash,
        row_count=record.row_count,
        column_count=record.column_count,
        created_at=ensure_utc(record.created_at),
        storage_path=record.storage_path,
        profile_path=record.profile_path,
    )


def _experiment(record: ExperimentRecord) -> Experiment:
    """Converte um registro ExperimentRecord na entidade de domínio Experiment.

    Args:
        record (ExperimentRecord): Registro de banco de dados do experimento.

    Returns:
        Experiment: Instância da entidade de domínio correspondente.
    """
    return Experiment(
        id=record.id,
        name=record.name,
        description=record.description,
        mode=ExperimentMode(record.mode),
        status=ExperimentStatus(record.status),
        dataset_id=record.dataset_id,
        parent_experiment_id=record.parent_experiment_id,
        objective=record.objective,
        spec_version=record.spec_version,
        spec_json=record.experiment_spec_json,
        created_at=ensure_utc(record.created_at),
        updated_at=ensure_utc(record.updated_at),
        deleted_at=ensure_utc(record.deleted_at) if record.deleted_at else None,
    )


def _model_run(record: ExperimentRunRecord) -> ModelRun:
    """Converte um registro ExperimentRunRecord na entidade de domínio ModelRun.

    Args:
        record (ExperimentRunRecord): Registro de banco de dados da execução de modelo.

    Returns:
        ModelRun: Instância da entidade de domínio correspondente.
    """
    return ModelRun(
        id=record.id,
        experiment_id=record.experiment_id,
        model_id=record.model_id,
        run_index=record.run_index,
        status=ModelRunStatus(record.status),
        started_at=ensure_utc(record.started_at) if record.started_at else None,
        finished_at=ensure_utc(record.finished_at) if record.finished_at else None,
        duration_seconds=record.duration_seconds,
        random_seed=record.random_seed,
        parameters_json=record.parameters_json,
        metrics_json=record.metrics_json,
        artifact_path=record.artifact_path,
        error_message=record.error_message,
        execution_id=record.execution_id,
    )


def _tuning_trial(record: TuningTrialRecord) -> TuningTrial:
    """Converte um registro TuningTrialRecord na entidade de domínio TuningTrial.

    Args:
        record (TuningTrialRecord): Registro de banco de dados do trial de tuning.

    Returns:
        TuningTrial: Instância da entidade de domínio correspondente.
    """
    return TuningTrial(
        id=record.id,
        run_id=record.run_id,
        trial_number=record.trial_number,
        parameters_json=record.parameters_json,
        metric_value=record.metric_value,
        status=TuningTrialStatus(record.status),
    )


def _experiment_strategy(record: ExperimentStrategyRecord) -> ExperimentStrategy:
    """Converte um registro ExperimentStrategyRecord na entidade ExperimentStrategy.

    Args:
        record (ExperimentStrategyRecord): Registro de banco de dados da estratégia.

    Returns:
        ExperimentStrategy: Instância da entidade de domínio correspondente.
    """
    return ExperimentStrategy(
        id=record.id,
        experiment_id=record.experiment_id,
        source=record.source,
        provider_id=record.provider_id,
        strategy_json=record.strategy_json,
        created_at=ensure_utc(record.created_at),
    )


def _decision_run(record: DecisionRunRecord) -> DecisionRun:
    """Converte um registro DecisionRunRecord na entidade de domínio DecisionRun.

    Args:
        record (DecisionRunRecord): Registro de banco de dados da execução de decisão.

    Returns:
        DecisionRun: Instância da entidade de domínio correspondente.
    """
    return DecisionRun(
        id=record.id,
        dataset_id=record.dataset_id,
        experiment_id=record.experiment_id,
        provider_id=record.provider_id,
        requested_model=record.requested_model,
        resolved_model=record.resolved_model,
        decision_spec_version=record.decision_spec_version,
        question_spec_hash=record.question_spec_hash,
        spec_json=record.spec_json,
        status=DecisionRunStatus(record.status),
        row_count=record.row_count,
        processed_rows=record.processed_rows,
        request_count=record.request_count,
        cache_hits=record.cache_hits,
        input_tokens=record.input_tokens,
        estimated_cost=record.estimated_cost,
        latency_ms_total=record.latency_ms_total,
        artifact_path=record.artifact_path,
        calibration_run_id=record.calibration_run_id,
        error_message=record.error_message,
        created_at=ensure_utc(record.created_at),
        started_at=ensure_utc(record.started_at) if record.started_at else None,
        finished_at=ensure_utc(record.finished_at) if record.finished_at else None,
    )


def _calibration_run(record: CalibrationRunRecord) -> CalibrationRun:
    """Converte um registro CalibrationRunRecord na entidade CalibrationRun.

    Args:
        record (CalibrationRunRecord): Registro de banco de dados da calibração.

    Returns:
        CalibrationRun: Instância da entidade de domínio correspondente.
    """
    return CalibrationRun(
        id=record.id,
        decision_run_id=record.decision_run_id,
        task=record.task,
        metrics_json=record.metrics_json,
        artifact_path=record.artifact_path,
        created_at=ensure_utc(record.created_at),
    )


def _semantic_feature_set(record: SemanticFeatureSetRecord) -> SemanticFeatureSet:
    """Converte SemanticFeatureSetRecord na entidade SemanticFeatureSet.

    Args:
        record (SemanticFeatureSetRecord): Registro de features semânticas.

    Returns:
        SemanticFeatureSet: Instância da entidade de domínio correspondente.
    """
    return SemanticFeatureSet(
        id=record.id,
        dataset_id=record.dataset_id,
        source_experiment_id=record.source_experiment_id,
        name=record.name,
        version=record.version,
        status=DecisionRunStatus(record.status),
        discovery_mode=SemanticDiscoveryMode(record.discovery_mode),
        questions_frozen=bool(record.questions_frozen),
        question_spec_json=record.question_spec_json,
        provider_id=record.provider_id,
        requested_model=record.requested_model,
        resolved_model=record.resolved_model,
        row_count=record.row_count,
        processed_rows=record.processed_rows,
        request_count=record.request_count,
        cache_hits=record.cache_hits,
        input_tokens=record.input_tokens,
        estimated_cost=record.estimated_cost,
        latency_ms_total=record.latency_ms_total,
        artifact_path=record.artifact_path,
        impact_json=record.impact_json,
        error_message=record.error_message,
        created_at=ensure_utc(record.created_at),
        started_at=ensure_utc(record.started_at) if record.started_at else None,
        finished_at=ensure_utc(record.finished_at) if record.finished_at else None,
    )


def _cascade_run(record: CascadeRunRecord) -> CascadeRun:
    """Converte um registro CascadeRunRecord na entidade de domínio CascadeRun.

    Args:
        record (CascadeRunRecord): Registro de banco de dados da execução em cascata.

    Returns:
        CascadeRun: Instância da entidade de domínio correspondente.
    """
    return CascadeRun(
        id=record.id,
        decision_run_id=record.decision_run_id,
        policy_json=record.policy_json,
        metrics_json=record.metrics_json,
        artifact_path=record.artifact_path,
        created_at=ensure_utc(record.created_at),
    )


class SqlAlchemyDatasetRepository:
    """Implementa o repositório de datasets utilizando SQLAlchemy.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório com o gerenciador de banco de dados.

        Args:
            database (Database): Instância de conexão do banco.
        """
        self.database = database

    def add(self, dataset: Dataset) -> Dataset:
        """Adiciona e persiste um novo dataset no banco de dados.

        Args:
            dataset (Dataset): Instância da entidade Dataset a ser persistida.

        Returns:
            Dataset: Entidade de dataset persistida.
        """
        with self.database.session() as session:
            session.add(
                DatasetRecord(
                    id=dataset.id,
                    name=dataset.name,
                    original_filename=dataset.original_filename,
                    content_hash=dataset.content_hash,
                    row_count=dataset.row_count,
                    column_count=dataset.column_count,
                    created_at=dataset.created_at,
                    storage_path=dataset.storage_path,
                    profile_path=dataset.profile_path,
                )
            )
        return dataset

    def get(self, dataset_id: str) -> Dataset | None:
        """Recupera um dataset pelo seu identificador único.

        Args:
            dataset_id (str): Identificador único do dataset.

        Returns:
            Dataset | None: Entidade encontrada ou None se não existir.
        """
        with self.database.session() as session:
            record = session.get(DatasetRecord, dataset_id)
            return _dataset(record) if record else None

    def find_by_hash(self, content_hash: str) -> Dataset | None:
        """Busca um dataset pelo hash SHA-256 do seu conteúdo.

        Args:
            content_hash (str): Hash do conteúdo para verificação de duplicidade.

        Returns:
            Dataset | None: Entidade encontrada ou None se inexistente.
        """
        with self.database.session() as session:
            record = session.scalar(
                select(DatasetRecord).where(DatasetRecord.content_hash == content_hash)
            )
            return _dataset(record) if record else None

    def list_all(self) -> list[Dataset]:
        """Lista todos os datasets cadastrados ordenados por data de criação.

        Returns:
            list[Dataset]: Lista de entidades Dataset.
        """
        with self.database.session() as session:
            records = session.scalars(
                select(DatasetRecord).order_by(DatasetRecord.created_at)
            ).all()
            return [_dataset(record) for record in records]


class SqlAlchemyExperimentRepository:
    """Implementa o repositório de experimentos utilizando SQLAlchemy.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório com o gerenciador de banco de dados.

        Args:
            database (Database): Instância de conexão do banco.
        """
        self.database = database

    def add(self, experiment: Experiment) -> Experiment:
        """Persiste um novo experimento no banco de dados.

        Args:
            experiment (Experiment): Entidade do experimento a persistir.

        Returns:
            Experiment: Instância persistida do experimento.
        """
        with self.database.session() as session:
            session.add(
                ExperimentRecord(
                    id=experiment.id,
                    name=experiment.name,
                    description=experiment.description,
                    mode=experiment.mode.value,
                    status=experiment.status.value,
                    dataset_id=experiment.dataset_id,
                    parent_experiment_id=experiment.parent_experiment_id,
                    objective=experiment.objective,
                    spec_version=experiment.spec_version,
                    experiment_spec_json=experiment.spec_json,
                    created_at=experiment.created_at,
                    updated_at=experiment.updated_at,
                )
            )
        return experiment

    def get(self, experiment_id: str) -> Experiment | None:
        """Busca um experimento pelo identificador único.

        Args:
            experiment_id (str): Identificador único do experimento.

        Returns:
            Experiment | None: Entidade encontrada ou None se não existir.
        """
        with self.database.session() as session:
            record = session.get(ExperimentRecord, experiment_id)
            return _experiment(record) if record else None

    def list_all(self) -> list[Experiment]:
        """Lista todos os experimentos ativos (não excluídos logicamente).

        Returns:
            list[Experiment]: Lista de experimentos ativos ordenados por criação.
        """
        with self.database.session() as session:
            records = session.scalars(
                select(ExperimentRecord)
                .where(ExperimentRecord.deleted_at.is_(None))
                .order_by(ExperimentRecord.created_at)
            ).all()
            return [_experiment(record) for record in records]

    def replace_spec(
        self,
        experiment_id: str,
        *,
        spec_version: str,
        spec: Mapping[str, Any],
        name: str | None = None,
        dataset_id: str | None = None,
        objective: str | None = None,
    ) -> Experiment:
        """Substitui a especificação do experimento antes de sua primeira execução.

        Args:
            experiment_id (str): Identificador do experimento.
            spec_version (str): Versão da nova especificação.
            spec (Mapping[str, Any]): Dicionário com a nova especificação.
            name (str | None): Novo nome opcional.
            dataset_id (str | None): Novo identificador de dataset opcional.
            objective (str | None): Novo objetivo analítico opcional.

        Returns:
            Experiment: Instância atualizada do experimento.

        Raises:
            EntityNotFoundError: Se o experimento não for encontrado.
            ImmutableExperimentError: Se o experimento já foi executado e está bloqueado.
        """
        try:
            with self.database.session() as session:
                record = session.get(ExperimentRecord, experiment_id)
                if record is None:
                    raise EntityNotFoundError(f"Experiment {experiment_id} was not found")
                record.spec_version = spec_version
                record.experiment_spec_json = canonical_json(spec)
                if name is not None:
                    record.name = name
                if dataset_id is not None:
                    record.dataset_id = dataset_id
                if objective is not None:
                    record.objective = objective
                record.updated_at = utc_now()
                session.flush()
                result = _experiment(record)
        except IntegrityError as exc:
            if "experiment_config_immutable_after_first_run" in str(exc):
                raise ImmutableExperimentError(
                    "Executed experiment configuration cannot be changed; clone it instead"
                ) from exc
            raise
        return result

    def update_status(self, experiment_id: str, status: ExperimentStatus) -> Experiment:
        """Atualiza o status de ciclo de vida do experimento.

        Args:
            experiment_id (str): Identificador do experimento.
            status (ExperimentStatus): Novo status a atribuir.

        Returns:
            Experiment: Experimento com o status atualizado.

        Raises:
            EntityNotFoundError: Se o experimento não existir.
        """
        with self.database.session() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise EntityNotFoundError(f"Experiment {experiment_id} was not found")
            record.status = status.value
            record.updated_at = utc_now()
            session.flush()
            return _experiment(record)

    def rename(self, experiment_id: str, name: str) -> Experiment:
        """Renomeia um experimento ativo.

        Args:
            experiment_id (str): Identificador do experimento.
            name (str): Novo nome para o experimento.

        Returns:
            Experiment: Experimento renomeado.

        Raises:
            EntityNotFoundError: Se o experimento não existir ou estiver deletado.
        """
        with self.database.session() as session:
            record = session.scalar(
                select(ExperimentRecord).where(
                    ExperimentRecord.id == experiment_id,
                    ExperimentRecord.deleted_at.is_(None),
                )
            )
            if record is None:
                raise EntityNotFoundError(f"Experiment {experiment_id} was not found")
            record.name = name
            record.updated_at = utc_now()
            session.flush()
            return _experiment(record)

    def soft_delete(self, experiment_id: str) -> None:
        """Marca um experimento como excluído logicamente.

        Args:
            experiment_id (str): Identificador do experimento a excluir.

        Raises:
            EntityNotFoundError: Se o experimento não for encontrado ou já estiver excluído.
        """
        with self.database.session() as session:
            record = session.scalar(
                select(ExperimentRecord).where(
                    ExperimentRecord.id == experiment_id,
                    ExperimentRecord.deleted_at.is_(None),
                )
            )
            if record is None:
                raise EntityNotFoundError(f"Experiment {experiment_id} was not found")
            record.deleted_at = utc_now()
            record.updated_at = record.deleted_at


class SqlAlchemyModelRunRepository:
    """Implementa o repositório de execuções de modelos de ML com SQLAlchemy.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    _terminal_statuses = {
        ModelRunStatus.COMPLETED,
        ModelRunStatus.FAILED,
        ModelRunStatus.CANCELLED,
    }

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de model runs.

        Args:
            database (Database): Gerenciador de sessões de banco de dados.
        """
        self.database = database

    def create(
        self,
        *,
        experiment_id: str,
        model_id: str,
        random_seed: int,
        parameters: Mapping[str, Any],
        execution_id: str | None = None,
    ) -> ModelRun:
        """Cria e enfileira uma nova execução de modelo.

        Args:
            experiment_id (str): Identificador do experimento associado.
            model_id (str): Identificador do algoritmo de modelo.
            random_seed (int): Semente de aleatoriedade da execução.
            parameters (Mapping[str, Any]): Hiperparâmetros configurados.
            execution_id (str | None): Identificador de correlação de execução opcional.

        Returns:
            ModelRun: Instância da execução criada em estado QUEUED.
        """
        with self.database.session() as session:
            latest_index = session.scalar(
                select(func.max(ExperimentRunRecord.run_index)).where(
                    ExperimentRunRecord.experiment_id == experiment_id,
                    ExperimentRunRecord.model_id == model_id,
                )
            )
            record = ExperimentRunRecord(
                id=new_id(),
                experiment_id=experiment_id,
                model_id=model_id,
                run_index=(latest_index or 0) + 1,
                status=ModelRunStatus.QUEUED.value,
                random_seed=random_seed,
                parameters_json=canonical_json(parameters),
                execution_id=execution_id,
            )
            session.add(record)
            session.flush()
            return _model_run(record)

    def get(self, run_id: str) -> ModelRun | None:
        """Recupera uma execução de modelo pelo identificador.

        Args:
            run_id (str): Identificador único da execução.

        Returns:
            ModelRun | None: Execução encontrada ou None se não existir.
        """
        with self.database.session() as session:
            record = session.get(ExperimentRunRecord, run_id)
            return _model_run(record) if record else None

    def list_for_experiment(self, experiment_id: str) -> list[ModelRun]:
        """Lista todas as execuções de modelo de um experimento.

        Args:
            experiment_id (str): Identificador do experimento pai.

        Returns:
            list[ModelRun]: Lista de execuções do experimento ordenadas.
        """
        with self.database.session() as session:
            records = session.scalars(
                select(ExperimentRunRecord)
                .where(ExperimentRunRecord.experiment_id == experiment_id)
                .order_by(ExperimentRunRecord.model_id, ExperimentRunRecord.run_index)
            ).all()
            return [_model_run(record) for record in records]

    def transition(
        self,
        run_id: str,
        status: ModelRunStatus,
        *,
        metrics: Mapping[str, Any] | None = None,
        artifact_path: str | None = None,
        error_message: str | None = None,
    ) -> ModelRun:
        """Transiciona o status e atualiza métricas ou artefatos da execução de modelo.

        Args:
            run_id (str): Identificador da execução.
            status (ModelRunStatus): Novo status almejado.
            metrics (Mapping[str, Any] | None): Métricas computadas se finalizado.
            artifact_path (str | None): Caminho do artefato gerado.
            error_message (str | None): Mensagem de erro caso tenha falhado.

        Returns:
            ModelRun: Instância atualizada da execução.

        Raises:
            EntityNotFoundError: Se a execução não for localizada.
            ImmutableModelRunError: Se a execução já estiver em estado terminal.
        """
        with self.database.session() as session:
            record = session.get(ExperimentRunRecord, run_id)
            if record is None:
                raise EntityNotFoundError(f"Model run {run_id} was not found")
            if ModelRunStatus(record.status) in self._terminal_statuses:
                raise ImmutableModelRunError("Terminal model runs cannot be changed")

            now = utc_now()
            if status == ModelRunStatus.RUNNING and record.started_at is None:
                record.started_at = now
            if status in self._terminal_statuses:
                record.finished_at = now
                started_at = record.started_at or now
                record.duration_seconds = (now - ensure_utc(started_at)).total_seconds()

            record.status = status.value
            record.metrics_json = canonical_json(metrics) if metrics is not None else None
            record.artifact_path = artifact_path
            record.error_message = error_message
            session.flush()
            return _model_run(record)


class SqlAlchemyTuningTrialRepository:
    """Implementa o repositório de tentativas de hiperparametrização (tuning trials).

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de tuning trials.

        Args:
            database (Database): Gerenciador de sessões do banco.
        """
        self.database = database

    def add(self, trial: TuningTrial) -> TuningTrial:
        """Persiste uma tentativa de ajuste de hiperparâmetros.

        Args:
            trial (TuningTrial): Entidade do trial a ser persistido.

        Returns:
            TuningTrial: Instância persistida do trial.
        """
        with self.database.session() as session:
            session.add(
                TuningTrialRecord(
                    id=trial.id,
                    run_id=trial.run_id,
                    trial_number=trial.trial_number,
                    parameters_json=trial.parameters_json,
                    metric_value=trial.metric_value,
                    status=trial.status.value,
                )
            )
        return trial

    def list_for_run(self, run_id: str) -> list[TuningTrial]:
        """Lista todas as tentativas registradas para uma execução de modelo.

        Args:
            run_id (str): Identificador da execução de modelo.

        Returns:
            list[TuningTrial]: Lista de trials ordenados por número sequencial.
        """
        with self.database.session() as session:
            records = session.scalars(
                select(TuningTrialRecord)
                .where(TuningTrialRecord.run_id == run_id)
                .order_by(TuningTrialRecord.trial_number)
            ).all()
            return [_tuning_trial(record) for record in records]


class SqlAlchemySettingRepository:
    """Implementa o repositório de configurações de sistema (chave-valor).

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de configurações.

        Args:
            database (Database): Gerenciador de sessões de banco.
        """
        self.database = database

    def get(self, key: str) -> str | None:
        """Recupera o valor de uma configuração pela sua chave.

        Args:
            key (str): Chave única da configuração.

        Returns:
            str | None: Valor armazenado ou None se a chave não existir.
        """
        with self.database.session() as session:
            record = session.get(SettingRecord, key)
            return record.value if record else None

    def set(self, key: str, value: str) -> None:
        """Cria ou atualiza uma configuração no banco de dados.

        Args:
            key (str): Chave da configuração.
            value (str): Novo valor textual a ser armazenado.
        """
        with self.database.session() as session:
            record = session.get(SettingRecord, key)
            if record is None:
                session.add(SettingRecord(key=key, value=value, updated_at=utc_now()))
            else:
                record.value = value
                record.updated_at = utc_now()


class SqlAlchemyDecisionRunRepository:
    """Implementa o repositório de execuções de decisões LLM com persistência em SQLAlchemy.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    _terminal = {
        DecisionRunStatus.COMPLETED,
        DecisionRunStatus.FAILED,
        DecisionRunStatus.CANCELLED,
    }
    _checkpoint_fields = {
        "resolved_model",
        "processed_rows",
        "request_count",
        "cache_hits",
        "input_tokens",
        "estimated_cost",
        "latency_ms_total",
        "artifact_path",
        "calibration_run_id",
        "error_message",
    }

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de execuções de decisão.

        Args:
            database (Database): Gerenciador de sessões do banco.
        """
        self.database = database

    def add(self, run: DecisionRun) -> DecisionRun:
        """Persiste uma nova execução de decisão.

        Args:
            run (DecisionRun): Entidade de execução de decisão.

        Returns:
            DecisionRun: Instância persistida da execução.
        """
        with self.database.session() as session:
            session.add(
                DecisionRunRecord(
                    id=run.id,
                    dataset_id=run.dataset_id,
                    experiment_id=run.experiment_id,
                    provider_id=run.provider_id,
                    requested_model=run.requested_model,
                    resolved_model=run.resolved_model,
                    decision_spec_version=run.decision_spec_version,
                    question_spec_hash=run.question_spec_hash,
                    spec_json=run.spec_json,
                    status=run.status.value,
                    row_count=run.row_count,
                    processed_rows=run.processed_rows,
                    request_count=run.request_count,
                    cache_hits=run.cache_hits,
                    input_tokens=run.input_tokens,
                    estimated_cost=run.estimated_cost,
                    latency_ms_total=run.latency_ms_total,
                    artifact_path=run.artifact_path,
                    calibration_run_id=run.calibration_run_id,
                    error_message=run.error_message,
                    created_at=run.created_at,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                )
            )
        return run

    def get(self, run_id: str) -> DecisionRun | None:
        """Recupera uma execução de decisão pelo seu ID.

        Args:
            run_id (str): Identificador único da execução de decisão.

        Returns:
            DecisionRun | None: Entidade encontrada ou None se inexistente.
        """
        with self.database.session() as session:
            record = session.get(DecisionRunRecord, run_id)
            return _decision_run(record) if record else None

    def list_all(self) -> list[DecisionRun]:
        """Lista todas as execuções de decisão ordenadas pela data de criação decrescente.

        Returns:
            list[DecisionRun]: Lista de execuções de decisão.
        """
        with self.database.session() as session:
            records = session.scalars(
                select(DecisionRunRecord).order_by(DecisionRunRecord.created_at.desc())
            ).all()
            return [_decision_run(record) for record in records]

    def fail_interrupted(self) -> int:
        """Marca como FAILED execuções que ficaram ativas após reinício do processo.

        Returns:
            int: Quantidade de registros atualizados para FAILED.
        """
        with self.database.session() as session:
            result = session.execute(
                update(DecisionRunRecord)
                .where(
                    DecisionRunRecord.status.in_(
                        (DecisionRunStatus.QUEUED.value, DecisionRunStatus.RUNNING.value)
                    )
                )
                .values(
                    status=DecisionRunStatus.FAILED.value,
                    finished_at=utc_now(),
                    error_message="decision_job_interrupted",
                )
            )
            return result.rowcount

    def transition(
        self,
        run_id: str,
        status: DecisionRunStatus,
        **changes: Any,
    ) -> DecisionRun:
        """Transiciona o status e atualiza campos de progresso da execução de decisão.

        Args:
            run_id (str): Identificador da execução.
            status (DecisionRunStatus): Novo status a atribuir.
            **changes (Any): Campos opcionais de checkpoint a atualizar.

        Returns:
            DecisionRun: Entidade com o status atualizado.

        Raises:
            EntityNotFoundError: Se a execução não existir.
            ImmutableModelRunError: Se a execução já estiver em estado terminal.
            ValueError: Se houver campos não suportados em changes.
        """
        with self.database.session() as session:
            record = session.get(DecisionRunRecord, run_id)
            if record is None:
                raise EntityNotFoundError(f"Decision run {run_id} was not found")
            current = DecisionRunStatus(record.status)
            if current in self._terminal:
                raise ImmutableModelRunError("Terminal decision runs cannot be changed")
            self._apply_changes(record, changes)
            now = utc_now()
            if status == DecisionRunStatus.RUNNING and record.started_at is None:
                record.started_at = now
            if status in self._terminal:
                record.finished_at = now
            record.status = status.value
            session.flush()
            return _decision_run(record)

    def checkpoint(self, run_id: str, **changes: Any) -> DecisionRun:
        """Salva um checkpoint intermediário do progresso da execução.

        Args:
            run_id (str): Identificador da execução de decisão.
            **changes (Any): Campos de checkpoint a persistir.

        Returns:
            DecisionRun: Entidade atualizada com o checkpoint.

        Raises:
            EntityNotFoundError: Se a execução não existir.
            ImmutableModelRunError: Se a execução já estiver em estado terminal.
            ValueError: Se houver campos não permitidos em changes.
        """
        with self.database.session() as session:
            record = session.get(DecisionRunRecord, run_id)
            if record is None:
                raise EntityNotFoundError(f"Decision run {run_id} was not found")
            if DecisionRunStatus(record.status) in self._terminal:
                raise ImmutableModelRunError("Terminal decision runs cannot be changed")
            self._apply_changes(record, changes)
            session.flush()
            return _decision_run(record)

    def _apply_changes(self, record: DecisionRunRecord, changes: Mapping[str, Any]) -> None:
        """Aplica alterações validadas no registro da execução de decisão.

        Args:
            record (DecisionRunRecord): Registro SQLAlchemy a modificar.
            changes (Mapping[str, Any]): Dicionário com campos e valores a atualizar.

        Raises:
            ValueError: Se houver campos fora de _checkpoint_fields.
        """
        unknown = set(changes) - self._checkpoint_fields
        if unknown:
            raise ValueError(f"Unsupported decision run fields: {sorted(unknown)}")
        for key, value in changes.items():
            setattr(record, key, value)


class SqlAlchemyCalibrationRunRepository:
    """Implementa o repositório de calibração de decisões em SQLAlchemy.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de calibrações.

        Args:
            database (Database): Gerenciador de sessões de banco.
        """
        self.database = database

    def add(self, run: CalibrationRun) -> CalibrationRun:
        """Persiste uma nova execução de calibração.

        Args:
            run (CalibrationRun): Entidade de calibração a persistir.

        Returns:
            CalibrationRun: Instância persistida da calibração.
        """
        with self.database.session() as session:
            session.add(
                CalibrationRunRecord(
                    id=run.id,
                    decision_run_id=run.decision_run_id,
                    task=run.task,
                    metrics_json=run.metrics_json,
                    artifact_path=run.artifact_path,
                    created_at=run.created_at,
                )
            )
        return run

    def get(self, run_id: str) -> CalibrationRun | None:
        """Recupera uma calibração pelo seu identificador único.

        Args:
            run_id (str): Identificador único da calibração.

        Returns:
            CalibrationRun | None: Entidade encontrada ou None se inexistente.
        """
        with self.database.session() as session:
            record = session.get(CalibrationRunRecord, run_id)
            return _calibration_run(record) if record else None

    def get_for_decision_run(self, decision_run_id: str) -> CalibrationRun | None:
        """Recupera a calibração vinculada a uma execução de decisão.

        Args:
            decision_run_id (str): Identificador da execução de decisão pai.

        Returns:
            CalibrationRun | None: Calibração vinculada ou None se inexistente.
        """
        with self.database.session() as session:
            record = session.scalar(
                select(CalibrationRunRecord).where(
                    CalibrationRunRecord.decision_run_id == decision_run_id
                )
            )
            return _calibration_run(record) if record else None


class SqlAlchemyDecisionCacheRepository:
    """Implementa o repositório de cache de respostas LLM por linha e contexto.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de cache de decisões.

        Args:
            database (Database): Gerenciador de sessões de banco.
        """
        self.database = database

    def get(self, context_key: str) -> Mapping[str, Any] | None:
        """Recupera resposta do cache pela chave de contexto.

        Args:
            context_key (str): Chave identificadora única do contexto da predição.

        Returns:
            Mapping[str, Any] | None: Resposta JSON deserializada ou None se cache miss.
        """
        with self.database.session() as session:
            record = session.get(DecisionCacheRecord, context_key)
            return json.loads(record.response_json) if record else None

    def put(
        self,
        *,
        context_key: str,
        cache_key: str,
        dataset_hash: str,
        row_hash: str,
        question_spec_hash: str,
        provider_id: str,
        requested_model: str,
        resolved_model: str,
        response: Mapping[str, Any],
    ) -> None:
        """Armazena uma resposta no cache de decisões de forma idempotente.

        Args:
            context_key (str): Chave única composta do contexto.
            cache_key (str): Chave semântica do prompt/pergunta.
            dataset_hash (str): Hash do dataset.
            row_hash (str): Hash da linha do dataset.
            question_spec_hash (str): Hash da especificação da pergunta.
            provider_id (str): Identificador do provedor LLM.
            requested_model (str): Nome do modelo solicitado.
            resolved_model (str): Nome do modelo resolvido.
            response (Mapping[str, Any]): Dicionário com a resposta a armazenar.
        """
        with self.database.session() as session:
            existing = session.get(DecisionCacheRecord, context_key)
            if existing is not None:
                return
            session.add(
                DecisionCacheRecord(
                    context_key=context_key,
                    cache_key=cache_key,
                    dataset_hash=dataset_hash,
                    row_hash=row_hash,
                    question_spec_hash=question_spec_hash,
                    provider_id=provider_id,
                    requested_model=requested_model,
                    resolved_model=resolved_model,
                    response_json=canonical_json(response),
                    created_at=utc_now(),
                )
            )


class SqlAlchemySemanticFeatureSetRepository:
    """Implementa o repositório de conjuntos de features semânticas.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    _terminal = {
        DecisionRunStatus.COMPLETED,
        DecisionRunStatus.FAILED,
        DecisionRunStatus.CANCELLED,
    }
    _mutable_fields = {
        "resolved_model",
        "processed_rows",
        "request_count",
        "cache_hits",
        "input_tokens",
        "estimated_cost",
        "latency_ms_total",
        "artifact_path",
        "impact_json",
        "error_message",
    }

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de conjuntos de features semânticas.

        Args:
            database (Database): Gerenciador de sessões de banco.
        """
        self.database = database

    def add(self, feature_set: SemanticFeatureSet) -> SemanticFeatureSet:
        """Persiste um novo conjunto de features semânticas.

        Args:
            feature_set (SemanticFeatureSet): Conjunto de features semânticas a persistir.

        Returns:
            SemanticFeatureSet: Entidade persistida.
        """
        with self.database.session() as session:
            session.add(
                SemanticFeatureSetRecord(
                    id=feature_set.id,
                    dataset_id=feature_set.dataset_id,
                    source_experiment_id=feature_set.source_experiment_id,
                    name=feature_set.name,
                    version=feature_set.version,
                    status=feature_set.status.value,
                    discovery_mode=feature_set.discovery_mode.value,
                    questions_frozen=int(feature_set.questions_frozen),
                    question_spec_json=feature_set.question_spec_json,
                    provider_id=feature_set.provider_id,
                    requested_model=feature_set.requested_model,
                    resolved_model=feature_set.resolved_model,
                    row_count=feature_set.row_count,
                    processed_rows=feature_set.processed_rows,
                    request_count=feature_set.request_count,
                    cache_hits=feature_set.cache_hits,
                    input_tokens=feature_set.input_tokens,
                    estimated_cost=feature_set.estimated_cost,
                    latency_ms_total=feature_set.latency_ms_total,
                    artifact_path=feature_set.artifact_path,
                    impact_json=feature_set.impact_json,
                    error_message=feature_set.error_message,
                    created_at=feature_set.created_at,
                    started_at=feature_set.started_at,
                    finished_at=feature_set.finished_at,
                )
            )
        return feature_set

    def get(self, feature_set_id: str) -> SemanticFeatureSet | None:
        """Recupera um conjunto de features semânticas pelo identificador.

        Args:
            feature_set_id (str): Identificador único do conjunto.

        Returns:
            SemanticFeatureSet | None: Entidade encontrada ou None se não existir.
        """
        with self.database.session() as session:
            record = session.get(SemanticFeatureSetRecord, feature_set_id)
            return _semantic_feature_set(record) if record else None

    def list_all(self) -> list[SemanticFeatureSet]:
        """Lista todos os conjuntos de features semânticas ordenados por data decrescente.

        Returns:
            list[SemanticFeatureSet]: Lista de entidades encontradas.
        """
        with self.database.session() as session:
            records = session.scalars(
                select(SemanticFeatureSetRecord).order_by(
                    SemanticFeatureSetRecord.created_at.desc()
                )
            ).all()
            return [_semantic_feature_set(record) for record in records]

    def transition(
        self, feature_set_id: str, status: DecisionRunStatus, **changes: Any
    ) -> SemanticFeatureSet:
        """Transiciona o status e atualiza métricas de execução do conjunto semântico.

        Args:
            feature_set_id (str): Identificador do conjunto de features.
            status (DecisionRunStatus): Novo status a atribuir.
            **changes (Any): Campos modificáveis a atualizar.

        Returns:
            SemanticFeatureSet: Entidade atualizada.

        Raises:
            EntityNotFoundError: Se o conjunto não for localizado.
            ImmutableModelRunError: Se o conjunto já estiver em estado terminal.
            ValueError: Se houver campos inválidos em changes.
        """
        with self.database.session() as session:
            record = self._record(session, feature_set_id)
            if DecisionRunStatus(record.status) in self._terminal:
                raise ImmutableModelRunError("Terminal semantic feature sets cannot be changed")
            self._apply(record, changes)
            now = utc_now()
            if status == DecisionRunStatus.RUNNING and record.started_at is None:
                record.started_at = now
            if status in self._terminal:
                record.finished_at = now
            record.status = status.value
            session.flush()
            return _semantic_feature_set(record)

    def checkpoint(self, feature_set_id: str, **changes: Any) -> SemanticFeatureSet:
        """Atualiza checkpoint de progresso do processamento de features semânticas.

        Args:
            feature_set_id (str): Identificador do conjunto.
            **changes (Any): Campos modificáveis com o progresso intermediário.

        Returns:
            SemanticFeatureSet: Entidade atualizada.

        Raises:
            EntityNotFoundError: Se o conjunto não for encontrado.
            ImmutableModelRunError: Se o conjunto já estiver em estado terminal.
            ValueError: Se houver campos não permitidos.
        """
        with self.database.session() as session:
            record = self._record(session, feature_set_id)
            if DecisionRunStatus(record.status) in self._terminal:
                raise ImmutableModelRunError("Terminal semantic feature sets cannot be changed")
            self._apply(record, changes)
            session.flush()
            return _semantic_feature_set(record)

    @staticmethod
    def _record(session: Any, feature_set_id: str) -> SemanticFeatureSetRecord:
        """Busca o registro SQLAlchemy do conjunto ou levanta erro de não encontrado.

        Args:
            session (Any): Sessão ativa do banco.
            feature_set_id (str): Identificador do conjunto.

        Returns:
            SemanticFeatureSetRecord: Registro SQLAlchemy encontrado.

        Raises:
            EntityNotFoundError: Se o registro não for localizado.
        """
        record = session.get(SemanticFeatureSetRecord, feature_set_id)
        if record is None:
            raise EntityNotFoundError(f"Semantic feature set {feature_set_id} was not found")
        return record

    def _apply(self, record: SemanticFeatureSetRecord, changes: Mapping[str, Any]) -> None:
        """Aplica mudanças validadas no registro de features semânticas.

        Args:
            record (SemanticFeatureSetRecord): Registro SQLAlchemy a alterar.
            changes (Mapping[str, Any]): Chaves e valores a modificar.

        Raises:
            ValueError: Se alguma chave não constar em _mutable_fields.
        """
        unknown = set(changes) - self._mutable_fields
        if unknown:
            raise ValueError(f"Unsupported semantic feature fields: {sorted(unknown)}")
        for key, value in changes.items():
            setattr(record, key, value)


class SqlAlchemyCascadeRunRepository:
    """Implementa o repositório de execuções em cascata (confidence cascade).

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de cascade runs.

        Args:
            database (Database): Gerenciador de sessões do banco.
        """
        self.database = database

    def add(self, run: CascadeRun) -> CascadeRun:
        """Persiste uma nova execução em cascata.

        Args:
            run (CascadeRun): Entidade de execução em cascata.

        Returns:
            CascadeRun: Instância persistida da cascata.
        """
        with self.database.session() as session:
            session.add(
                CascadeRunRecord(
                    id=run.id,
                    decision_run_id=run.decision_run_id,
                    policy_json=run.policy_json,
                    metrics_json=run.metrics_json,
                    artifact_path=run.artifact_path,
                    created_at=run.created_at,
                )
            )
        return run

    def get(self, run_id: str) -> CascadeRun | None:
        """Recupera uma execução em cascata pelo identificador único.

        Args:
            run_id (str): Identificador único da execução.

        Returns:
            CascadeRun | None: Entidade encontrada ou None se não existir.
        """
        with self.database.session() as session:
            record = session.get(CascadeRunRecord, run_id)
            return _cascade_run(record) if record else None

    def list_all(self) -> list[CascadeRun]:
        """Lista todas as execuções em cascata ordenadas por criação decrescente.

        Returns:
            list[CascadeRun]: Lista de execuções em cascata.
        """
        with self.database.session() as session:
            records = session.scalars(
                select(CascadeRunRecord).order_by(CascadeRunRecord.created_at.desc())
            ).all()
            return [_cascade_run(record) for record in records]


class SqlAlchemyExperimentStrategyRepository:
    """Implementa o repositório de estratégias de recomendação de experimentos.

    Attributes:
        database (Database): Gerenciador de conexão e sessões do banco de dados.
    """

    def __init__(self, database: Database) -> None:
        """Inicializa o repositório de estratégias.

        Args:
            database (Database): Gerenciador de sessões de banco.
        """
        self.database = database

    def add(self, strategy: ExperimentStrategy) -> ExperimentStrategy:
        """Persiste uma auditoria de estratégia gerada pelo advisor.

        Args:
            strategy (ExperimentStrategy): Entidade de estratégia a persistir.

        Returns:
            ExperimentStrategy: Instância persistida da estratégia.
        """
        with self.database.session() as session:
            session.add(
                ExperimentStrategyRecord(
                    id=strategy.id,
                    experiment_id=strategy.experiment_id,
                    source=strategy.source,
                    provider_id=strategy.provider_id,
                    strategy_json=strategy.strategy_json,
                    created_at=strategy.created_at,
                )
            )
        return strategy

    def add_experiment_with_strategy(
        self,
        experiment: Experiment,
        strategy: ExperimentStrategy,
    ) -> Experiment:
        """Persiste o experimento rascunho e sua estratégia gerada em uma mesma transação.

        Args:
            experiment (Experiment): Entidade do experimento rascunho.
            strategy (ExperimentStrategy): Entidade da estratégia do Advisor correspondente.

        Returns:
            Experiment: Instância persistida do experimento.

        Raises:
            ValueError: Se a estratégia não referenciar o mesmo experiment_id.
        """
        if strategy.experiment_id != experiment.id:
            raise ValueError("Strategy must reference the experiment being persisted")
        with self.database.session() as session:
            session.add(
                ExperimentRecord(
                    id=experiment.id,
                    name=experiment.name,
                    description=experiment.description,
                    mode=experiment.mode.value,
                    status=experiment.status.value,
                    dataset_id=experiment.dataset_id,
                    parent_experiment_id=experiment.parent_experiment_id,
                    objective=experiment.objective,
                    spec_version=experiment.spec_version,
                    experiment_spec_json=experiment.spec_json,
                    created_at=experiment.created_at,
                    updated_at=experiment.updated_at,
                )
            )
            session.add(
                ExperimentStrategyRecord(
                    id=strategy.id,
                    experiment_id=strategy.experiment_id,
                    source=strategy.source,
                    provider_id=strategy.provider_id,
                    strategy_json=strategy.strategy_json,
                    created_at=strategy.created_at,
                )
            )
        return experiment

    def get_for_experiment(self, experiment_id: str) -> ExperimentStrategy | None:
        """Recupera a auditoria de estratégia vinculada a um experimento.

        Args:
            experiment_id (str): Identificador do experimento.

        Returns:
            ExperimentStrategy | None: Estratégia associada ou None se inexistente.
        """
        with self.database.session() as session:
            record = session.scalar(
                select(ExperimentStrategyRecord).where(
                    ExperimentStrategyRecord.experiment_id == experiment_id
                )
            )
            return _experiment_strategy(record) if record else None
