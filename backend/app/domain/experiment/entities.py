from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.domain.common import canonical_json, json_object, new_id, utc_now


class ExperimentMode(StrEnum):
    """Representa o modo de configuração e execução de um experimento."""

    AUTO = "auto"
    ASSISTED = "assisted"
    EXPERT = "expert"


class ExperimentStatus(StrEnum):
    """Representa a situação do ciclo de vida de um experimento."""

    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelRunStatus(StrEnum):
    """Representa o estado de execução do treinamento de um modelo."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TuningTrialStatus(StrEnum):
    """Representa o estado de uma tentativa de sintonia de hiperparâmetros."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Experiment:
    """Representa um experimento de aprendizado de máquina na plataforma.

    Attributes:
        name (str): Nome amigável de exibição do experimento.
        mode (ExperimentMode): Modo de criação e operação do experimento.
        dataset_id (str): Identificador do dataset associado.
        objective (str): Descrição do objetivo analítico informado.
        spec_version (str): Versão da especificação técnica utilizada.
        spec_json (str): Especificação do experimento serializada em JSON.
        description (str | None): Detalhamento opcional do experimento.
        parent_experiment_id (str | None): Identificador do experimento pai se derivado.
        status (ExperimentStatus): Situação atual de execução do experimento.
        id (str): Identificador único universal do experimento.
        created_at (datetime): Timestamp de criação do registro em UTC.
        updated_at (datetime): Timestamp da última modificação em UTC.
        deleted_at (datetime | None): Timestamp de exclusão lógica ou None se ativo.
    """

    name: str
    mode: ExperimentMode
    dataset_id: str
    objective: str
    spec_version: str
    spec_json: str
    description: str | None = None
    parent_experiment_id: str | None = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        """Valida os invariantes de inicialização e conformidade da especificação.

        Raises:
            ValueError: Se nome, objetivo ou versão da spec estiverem vazios.
        """
        if not self.name.strip() or not self.objective.strip():
            raise ValueError("Experiment name and objective cannot be empty")
        if not self.spec_version.strip():
            raise ValueError("Experiment spec version cannot be empty")
        json_object(self.spec_json)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        mode: ExperimentMode,
        dataset_id: str,
        objective: str,
        spec_version: str,
        spec: Mapping[str, Any],
        description: str | None = None,
        parent_experiment_id: str | None = None,
    ) -> "Experiment":
        """Instancia um novo experimento serializando o dicionário de especificação.

        Args:
            name (str): Nome de exibição do experimento.
            mode (ExperimentMode): Modo de criação do experimento.
            dataset_id (str): Identificador do dataset base.
            objective (str): Objetivo preditivo do experimento.
            spec_version (str): Versão do contrato da especificação.
            spec (Mapping[str, Any]): Dicionário com a configuração técnica.
            description (str | None): Descrição adicional opcional.
            parent_experiment_id (str | None): Identificador de origem caso seja clone.

        Returns:
            Experiment: Nova instância do experimento configurado.
        """
        return cls(
            name=name,
            mode=mode,
            dataset_id=dataset_id,
            objective=objective,
            spec_version=spec_version,
            spec_json=canonical_json(spec),
            description=description,
            parent_experiment_id=parent_experiment_id,
        )

    @property
    def spec(self) -> dict[str, Any]:
        """Desserializa e retorna o dicionário com a especificação do experimento.

        Returns:
            dict[str, Any]: Estrutura com os parâmetros técnicos do experimento.
        """
        return json_object(self.spec_json)

    def clone(self, *, name: str | None = None) -> "Experiment":
        """Gera uma cópia do experimento em estado de rascunho com novo ID.

        Args:
            name (str | None): Novo nome para o experimento clonado ou None para manter.

        Returns:
            Experiment: Cópia isolada do experimento pronta para edição.
        """
        now = utc_now()
        return replace(
            self,
            id=new_id(),
            name=name or self.name,
            parent_experiment_id=self.id,
            status=ExperimentStatus.DRAFT,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )


@dataclass(frozen=True, slots=True)
class ModelRun:
    """Representa a execução de treinamento e avaliação de um modelo específico.

    Attributes:
        experiment_id (str): Identificador do experimento vinculado.
        model_id (str): Identificador do algoritmo de modelo no catálogo.
        run_index (int): Índice numérico sequencial da execução.
        status (ModelRunStatus): Estado atual de execução do modelo.
        random_seed (int): Semente de aleatoriedade configurada.
        parameters_json (str): Hiperparâmetros de treinamento serializados em JSON.
        id (str): Identificador único universal da execução.
        started_at (datetime | None): Data e hora de início do processamento.
        finished_at (datetime | None): Data e hora de término do processamento.
        duration_seconds (float | None): Duração total em segundos.
        metrics_json (str | None): Métricas computadas serializadas em JSON.
        artifact_path (str | None): Caminho do modelo treinado persistido em disco.
        error_message (str | None): Mensagem detalhada em caso de falha.
        execution_id (str | None): Identificador da rodada de execução em lote.
    """

    experiment_id: str
    model_id: str
    run_index: int
    status: ModelRunStatus
    random_seed: int
    parameters_json: str
    id: str = field(default_factory=new_id)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    metrics_json: str | None = None
    artifact_path: str | None = None
    error_message: str | None = None
    execution_id: str | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        """Desserializa e retorna os hiperparâmetros configurados para a execução.

        Returns:
            dict[str, Any]: Mapeamento de hiperparâmetros do modelo.
        """
        return json_object(self.parameters_json)

    @property
    def metrics(self) -> dict[str, Any] | None:
        """Desserializa e retorna as métricas avaliadas caso a execução esteja concluída.

        Returns:
            dict[str, Any] | None: Dicionário de métricas computadas ou None.
        """
        return json_object(self.metrics_json) if self.metrics_json else None


@dataclass(frozen=True, slots=True)
class TuningTrial:
    """Representa uma tentativa individual no processo de sintonia de hiperparâmetros.

    Attributes:
        run_id (str): Identificador da execução de modelo vinculada.
        trial_number (int): Número sequencial da tentativa.
        parameters_json (str): Conjunto de hiperparâmetros avaliados em JSON.
        status (TuningTrialStatus): Estado de finalização da tentativa.
        metric_value (float | None): Valor numérico da métrica alvo obtida.
        id (str): Identificador único universal da tentativa.
    """

    run_id: str
    trial_number: int
    parameters_json: str
    status: TuningTrialStatus
    metric_value: float | None = None
    id: str = field(default_factory=new_id)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        trial_number: int,
        parameters: Mapping[str, Any],
        status: TuningTrialStatus = TuningTrialStatus.PENDING,
        metric_value: float | None = None,
    ) -> "TuningTrial":
        """Instancia uma nova tentativa serializando os parâmetros fornecidos.

        Args:
            run_id (str): Identificador da execução do modelo.
            trial_number (int): Índice da tentativa no estudo.
            parameters (Mapping[str, Any]): Parâmetros sugeridos pelo otimizador.
            status (TuningTrialStatus): Status inicial da tentativa.
            metric_value (float | None): Valor inicial da métrica se já conhecido.

        Returns:
            TuningTrial: Instância da tentativa de sintonia criada.
        """
        return cls(
            run_id=run_id,
            trial_number=trial_number,
            parameters_json=canonical_json(parameters),
            status=status,
            metric_value=metric_value,
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Desserializa e retorna os hiperparâmetros testados nesta tentativa.

        Returns:
            dict[str, Any]: Hiperparâmetros avaliados.
        """
        return json_object(self.parameters_json)


@dataclass(frozen=True, slots=True)
class ExperimentStrategy:
    """Representa uma proposta de estratégia de modelagem associada a um experimento.

    Attributes:
        experiment_id (str): Identificador do experimento ao qual a estratégia pertence.
        source (str): Origem da estratégia recomendada (ex: 'auto' ou 'assisted').
        strategy_json (str): Conteúdo estruturado da recomendação em JSON.
        provider_id (str | None): Identificador do provedor LLM quando assistido.
        id (str): Identificador único universal da estratégia.
        created_at (datetime): Timestamp de geração em UTC.
    """

    experiment_id: str
    source: str
    strategy_json: str
    provider_id: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        source: str,
        strategy: Mapping[str, Any],
        provider_id: str | None = None,
    ) -> "ExperimentStrategy":
        """Instancia uma nova proposta de estratégia validando sua fonte de origem.

        Args:
            experiment_id (str): Identificador do experimento alvo.
            source (str): Origem da estratégia ('auto' ou 'assisted').
            strategy (Mapping[str, Any]): Dados da proposta de estratégia.
            provider_id (str | None): Identificador opcional do provedor de IA.

        Returns:
            ExperimentStrategy: Instância de estratégia criada e serializada.

        Raises:
            ValueError: Se a fonte informada for diferente de 'auto' ou 'assisted'.
        """
        if source not in {"auto", "assisted"}:
            raise ValueError("Strategy source must be auto or assisted")
        return cls(
            experiment_id=experiment_id,
            source=source,
            strategy_json=canonical_json(strategy),
            provider_id=provider_id,
        )

    @property
    def strategy(self) -> dict[str, Any]:
        """Desserializa e retorna a estrutura da estratégia recomendada.

        Returns:
            dict[str, Any]: Conteúdo da estratégia gerada.
        """
        return json_object(self.strategy_json)
