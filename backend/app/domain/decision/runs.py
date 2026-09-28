from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.common import canonical_json, json_object, new_id, utc_now
from app.domain.decision.questions import ChoiceQuestionSpec, NoulQuestionSpec
from app.domain.experiment.spec import AveragingStrategy, TaskType


class DecisionRunStatus(StrEnum):
    """Representa a situação de execução de um benchmark de decisão zero-shot."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ZeroShotDecisionSpec(BaseModel):
    """Define o contrato imutável e auditável para uma avaliação zero-shot de decisão.

    Attributes:
        spec_version (Literal["1.0"]): Versão do contrato da especificação de decisão.
        dataset_id (str): Identificador do dataset sob avaliação.
        target (str): Coluna dependente a ser inferida.
        task (TaskType): Tarefa de classificação supervisionada (binária ou multiclasse).
        positive_class (str | None): Rótulo positivo para classificação binária.
        input_columns (list[str]): Colunas do dataset encaminhadas no contexto da pergunta.
        question (DecisionQuestionSpec): Pergunta estruturada formulada.
        provider (str): Identificador do provedor de decisão selecionado.
        model (str): Nome do modelo remoto solicitado.
        averaging (AveragingStrategy): Estratégia de média para métricas multiclasse.
        evaluation_split (Literal["final_test"]): Partição avaliada nos testes.
        question_frozen (Literal[True]): Flag indicando congelamento da especificação da questão.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    task: Literal[
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    ]
    positive_class: str | None = None
    input_columns: list[str] = Field(min_length=1, max_length=100)
    question: NoulQuestionSpec | ChoiceQuestionSpec = Field(discriminator="type")
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    model: str = Field(min_length=1, max_length=256)
    averaging: AveragingStrategy = AveragingStrategy.MACRO
    evaluation_split: Literal["final_test"] = "final_test"
    question_frozen: Literal[True] = True

    @field_validator("input_columns")
    @classmethod
    def unique_inputs(cls, values: list[str]) -> list[str]:
        """Garante que as colunas de entrada informadas sejam limpas e únicas.

        Args:
            values (list[str]): Relação de colunas preditoras fornecidas.

        Returns:
            list[str]: Lista higienizada e sem duplicatas.

        Raises:
            ValueError: Se houver colunas em branco ou nomes repetidos.
        """
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("input columns must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("input columns must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_task_contract(self) -> "ZeroShotDecisionSpec":
        """Valida que o target não esteja nos inputs e a questão seja compatível com a tarefa.

        Returns:
            ZeroShotDecisionSpec: Própria instância validada.

        Raises:
            ValueError: Se o target estiver em input_columns ou a questão for incompatível.
        """
        if self.target in self.input_columns:
            raise ValueError("target cannot be sent as a decision input")
        if self.task == TaskType.BINARY_CLASSIFICATION:
            if not isinstance(self.question, NoulQuestionSpec):
                raise ValueError("binary classification requires a Noul question")
            if self.positive_class is None or not self.positive_class.strip():
                raise ValueError("binary classification requires positive_class")
        elif not isinstance(self.question, ChoiceQuestionSpec):
            raise ValueError("multiclass classification requires a Choice question")
        return self


@dataclass(frozen=True, slots=True)
class DecisionRun:
    """Representa a execução de um benchmark de decisão zero-shot contra um dataset.

    Attributes:
        dataset_id (str): Identificador do dataset avaliado.
        provider_id (str): Identificador do provedor utilizado.
        requested_model (str): Nome do modelo requisitado.
        decision_spec_version (str): Versão da especificação de decisão.
        question_spec_hash (str): Hash SHA-256 da especificação da questão.
        spec_json (str): Especificação serializada em formato JSON canônico.
        row_count (int): Total de linhas a processar.
        status (DecisionRunStatus): Estado de execução do benchmark.
        experiment_id (str | None): Experimento associado se executado em conjunto.
        resolved_model (str | None): Nome e versão exata do modelo retornado.
        processed_rows (int): Quantidade de linhas já processadas com sucesso.
        request_count (int): Número total de chamadas remotas de inferência.
        cache_hits (int): Quantidade de inferências reaproveitadas do cache.
        input_tokens (int | None): Total de tokens de entrada utilizados.
        estimated_cost (float | None): Custo monetário acumulado em dólares.
        latency_ms_total (float | None): Somatório de latência em milissegundos.
        artifact_path (str | None): Caminho do arquivo com previsões detalhadas.
        calibration_run_id (str | None): Identificador da execução de calibração associada.
        error_message (str | None): Mensagem detalhada em caso de falha.
        id (str): Identificador único da execução.
        created_at (datetime): Data/hora de registro em UTC.
        started_at (datetime | None): Data/hora de início da execução.
        finished_at (datetime | None): Data/hora de término da execução.
    """

    dataset_id: str
    provider_id: str
    requested_model: str
    decision_spec_version: str
    question_spec_hash: str
    spec_json: str
    row_count: int
    status: DecisionRunStatus = DecisionRunStatus.QUEUED
    experiment_id: str | None = None
    resolved_model: str | None = None
    processed_rows: int = 0
    request_count: int = 0
    cache_hits: int = 0
    input_tokens: int | None = None
    estimated_cost: float | None = None
    latency_ms_total: float | None = None
    artifact_path: str | None = None
    calibration_run_id: str | None = None
    error_message: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def create(
        cls,
        specification: ZeroShotDecisionSpec,
        *,
        row_count: int,
        question_spec_hash: str,
        experiment_id: str | None = None,
    ) -> "DecisionRun":
        """Instancia uma nova execução de decisão serializando a especificação.

        Args:
            specification (ZeroShotDecisionSpec): Especificação técnica do benchmark.
            row_count (int): Quantidade de linhas tabulares a avaliar.
            question_spec_hash (str): Hash identificador da questão formulada.
            experiment_id (str | None): Identificador opcional do experimento vinculado.

        Returns:
            DecisionRun: Nova entidade de execução em fila.
        """
        return cls(
            dataset_id=specification.dataset_id,
            experiment_id=experiment_id,
            provider_id=specification.provider,
            requested_model=specification.model,
            decision_spec_version=specification.spec_version,
            question_spec_hash=question_spec_hash,
            spec_json=canonical_json(specification.model_dump(mode="json")),
            row_count=row_count,
        )

    @property
    def specification(self) -> ZeroShotDecisionSpec:
        """Desserializa e reconstitui a especificação técnica da execução de decisão.

        Returns:
            ZeroShotDecisionSpec: Especificação validada da execução.
        """
        return ZeroShotDecisionSpec.model_validate(json_object(self.spec_json))


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    """Representa a execução de cálculo de calibração associada a inferências de decisão.

    Attributes:
        decision_run_id (str): Identificador da execução de decisão de origem.
        task (str): Tipo da tarefa de calibração executada.
        metrics_json (str): Métricas de calibração e curvas serializadas em JSON.
        artifact_path (str): Caminho físico para persistência do resumo de calibração.
        id (str): Identificador único universal da calibração.
        created_at (datetime): Timestamp de criação em UTC.
    """

    decision_run_id: str
    task: str
    metrics_json: str
    artifact_path: str
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        decision_run_id: str,
        task: str,
        metrics: Mapping[str, Any],
        artifact_path: str,
    ) -> "CalibrationRun":
        """Instancia um registro de calibração serializando as métricas.

        Args:
            decision_run_id (str): Identificador da execução de decisão base.
            task (str): Identificador da tarefa analítica associada.
            metrics (Mapping[str, Any]): Dicionário com métricas de calibração.
            artifact_path (str): Caminho para persistência do artefato.

        Returns:
            CalibrationRun: Instância de calibração criada.
        """
        return cls(
            decision_run_id=decision_run_id,
            task=task,
            metrics_json=canonical_json(metrics),
            artifact_path=artifact_path,
        )

    @property
    def metrics(self) -> dict[str, Any]:
        """Desserializa e retorna as métricas de calibração registradas.

        Returns:
            dict[str, Any]: Mapeamento contendo Brier score, ECE, histogramas e limiares.
        """
        return json_object(self.metrics_json)
