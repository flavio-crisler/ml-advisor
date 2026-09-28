from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.experiment.spec import AveragingStrategy, RocAucStrategy, TaskType


class ModelCapabilities(BaseModel):
    """Declara as capacidades e limitações operacionais de um modelo de Machine Learning.

    Attributes:
        supports_proba (bool): Indica se o modelo calcula probabilidades para classes.
        supports_imbalance_weighting (bool): Se aceita ponderação por pesos de classes.
        supports_categorical (bool): Se trata features categóricas nativamente.
        supports_gpu (bool): Se possui suporte a aceleração via GPU.
        requires_scaling (bool): Se necessita que features numéricas sejam normalizadas
            ou padronizadas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    supports_proba: bool = False
    supports_imbalance_weighting: bool = False
    supports_categorical: bool = False
    supports_gpu: bool = False
    requires_scaling: bool = False


class ModelMetadata(BaseModel):
    """Metadados descritivos e requisitos de um algoritmo de modelo cadastrado no catálogo.

    Attributes:
        id (str): Identificador único do modelo (ex: 'random_forest_classifier').
        display_name (str): Nome legível para exibição nas interfaces.
        tasks (frozenset[TaskType]): Conjunto de tarefas suportadas (ex: BINARY, REGRESSION).
        capabilities (ModelCapabilities): Metadados de capacidades operacionais.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    tasks: frozenset[TaskType]
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class MetricMetadata(BaseModel):
    """Metadados de uma métrica de avaliação cadastrada no catálogo.

    Attributes:
        id (str): Identificador da métrica (ex: 'f1', 'roc_auc', 'rmse').
        display_name (str): Nome legível para visualização.
        tasks (frozenset[TaskType]): Tarefas onde a métrica é aplicável.
        direction (Literal['maximize', 'minimize']): Direção de otimização da métrica.
        requires_proba (bool): Indica se necessita de probabilidades para cálculo.
        supported_averaging (tuple[AveragingStrategy, ...]): Estratégias de média para multiclasse.
        default_averaging (AveragingStrategy | None): Média padrão aplicada.
        roc_auc_strategies (tuple[RocAucStrategy, ...]): Estratégias ROC-AUC (ovr, ovo).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    tasks: frozenset[TaskType]
    direction: Literal["maximize", "minimize"]
    requires_proba: bool = False
    supported_averaging: tuple[AveragingStrategy, ...] = ()
    default_averaging: AveragingStrategy | None = None
    roc_auc_strategies: tuple[RocAucStrategy, ...] = ()


class PreprocessorMetadata(BaseModel):
    """Metadados declarativos de um pré-processador registrado no catálogo.

    Attributes:
        id (str): Identificador único do pré-processador.
        display_name (str): Nome amigável de exibição.
        category (str): Categoria da transformação (ex: 'numeric', 'missing').
        transformer_kind (Literal['feature', 'row', 'composite']): Escopo da transformação.
        strategies (tuple[str, ...]): Nomes das estratégias aceitas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    category: Literal[
        "missing",
        "categorical",
        "boolean",
        "numeric",
        "datetime",
        "outliers",
        "imbalance",
        "feature_selection",
        "text",
    ]
    transformer_kind: Literal["feature", "row", "composite"]
    strategies: tuple[str, ...]


class ValidationMetadata(BaseModel):
    """Metadados de estratégias de validação (divisão de treino/teste e cross-validation).

    Attributes:
        id (str): Identificador da estratégia (ex: 'stratified_kfold').
        display_name (str): Nome legível para interface.
        tasks (frozenset[TaskType]): Tarefas com as quais a validação é compatível.
        parameters (tuple[str, ...]): Parâmetros configuráveis da estratégia.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    tasks: frozenset[TaskType]
    parameters: tuple[str, ...]


class TunerMetadata(BaseModel):
    """Metadados de motores de otimização de hiperparâmetros (tuners).

    Attributes:
        id (str): Identificador único do tuner (ex: 'optuna').
        display_name (str): Nome de exibição.
        supports_pruning (bool): Se suporta interrupção antecipada de trials não promissores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    supports_pruning: bool
