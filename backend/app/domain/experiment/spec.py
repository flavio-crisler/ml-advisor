from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(StrEnum):
    """Representa a categoria da tarefa de aprendizado supervisionado."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class ColumnRole(StrEnum):
    """Representa o papel semântico atribuído a uma coluna do dataset."""

    FEATURE = "feature"
    TARGET = "target"
    ID = "id"
    IGNORED = "ignored"


class AveragingStrategy(StrEnum):
    """Representa a estratégia de agregação de métricas para problemas multiclasse."""

    MACRO = "macro"
    WEIGHTED = "weighted"


class RocAucStrategy(StrEnum):
    """Representa a estratégia de cálculo de curva ROC-AUC em cenários multiclasse."""

    OVR = "ovr"
    OVO = "ovo"


class ModelSpec(BaseModel):
    """Define a especificação e parametrização de um modelo a ser treinado.

    Attributes:
        id (str): Identificador do modelo registrado no catálogo.
        enabled (bool): Flag indicando se o modelo deve ser executado no experimento.
        parameters (dict[str, Any]): Hiperparâmetros específicos para instanciação do modelo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)


class PreprocessingStepSpec(BaseModel):
    """Define uma etapa sequencial no pipeline de pré-processamento de dados.

    Attributes:
        id (str): Identificador do transformador registrado no catálogo.
        strategy (str): Estratégia de transformação adotada para o componente.
        parameters (dict[str, Any]): Argumentos de configuração da transformação.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ValidationSpec(BaseModel):
    """Define o esquema de particionamento e validação estatística do modelo.

    Attributes:
        strategy (str): Estratégia de validação (ex: train_test_split, k_fold).
        test_size (float): Fração dos dados reservada para avaliação de teste.
        folds (int): Número de partições para validação cruzada.
        shuffle (bool): Se os dados devem ser embaralhados antes da partição.
        random_seed (int): Semente de aleatoriedade para reprodutibilidade.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str = Field(min_length=1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    folds: int = Field(default=5, ge=2, le=20)
    shuffle: bool = True
    random_seed: int = 41


class MetricsSpec(BaseModel):
    """Define as métricas de avaliação e regras de agregação do experimento.

    Attributes:
        primary (str): Métrica principal utilizada para ranking e seleção do melhor modelo.
        secondary (list[str]): Métricas adicionais monitoradas na avaliação.
        averaging (AveragingStrategy | None): Estratégia de média em tarefas multiclasse.
        roc_auc_strategy (RocAucStrategy | None): Abordagem One-vs-Rest ou One-vs-One.
        positive_label (str | int | float | bool | None): Rótulo da classe positiva em binária.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary: str = Field(min_length=1)
    secondary: list[str] = Field(default_factory=list)
    averaging: AveragingStrategy | None = None
    roc_auc_strategy: RocAucStrategy | None = None
    positive_label: str | int | float | bool | None = None


class TuningSpec(BaseModel):
    """Define a configuração do mecanismo de busca e sintonia de hiperparâmetros.

    Attributes:
        enabled (bool): Indica se a otimização automática de hiperparâmetros está habilitada.
        engine (str): Identificador do mecanismo de otimização (ex: 'optuna').
        trials (int): Limite máximo de tentativas de busca por modelo.
        metric (str | None): Identificador da métrica a ser otimizada.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    engine: str = "optuna"
    trials: int = Field(default=50, ge=1, le=1000)
    metric: str | None = None


class ExperimentSpec(BaseModel):
    """Representa a especificação técnica completa, determinística e reproduzível de um experimento.

    Attributes:
        spec_version (Literal["1.0"]): Versão do esquema estrutural da especificação.
        dataset_id (str): Identificador único do dataset associado.
        dataset_hash (str): Hash criptográfico do conteúdo do dataset para auditoria.
        profile_version (str): Versão do perfil de dados utilizado.
        name (str): Nome amigável de identificação do experimento.
        objective (str): Objetivo preditivo do experimento.
        task (TaskType): Tipo de problema de ML supervisionado.
        target (str): Nome da coluna definida como variável dependente.
        features (list[str]): Relação de colunas preditoras independentes.
        column_roles (dict[str, ColumnRole]): Mapeamento explícito do papel de cada coluna.
        models (list[ModelSpec]): Coleção de modelos e parâmetros a serem avaliados.
        preprocessing (list[PreprocessingStepSpec]): Sequência de transformações tabulares.
        validation (ValidationSpec): Critérios de particionamento e validação dos dados.
        metrics (MetricsSpec): Métricas primária e secundárias de avaliação.
        tuning (TuningSpec): Instruções para ajuste de hiperparâmetros.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=2000)
    task: TaskType
    target: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    column_roles: dict[str, ColumnRole]
    models: list[ModelSpec] = Field(min_length=1)
    preprocessing: list[PreprocessingStepSpec] = Field(default_factory=list)
    validation: ValidationSpec
    metrics: MetricsSpec
    tuning: TuningSpec = Field(default_factory=TuningSpec)

    @model_validator(mode="after")
    def validate_contract(self) -> "ExperimentSpec":
        """Valida a integridade lógica e os invariantes de contrato da especificação.

        Returns:
            ExperimentSpec: Própria instância validada.

        Raises:
            ValueError: Se houver sobreposição entre target e features, papéis inconsistentes
                ou métricas ausentes para tarefas multiclasse e binárias.
        """
        if len(self.features) != len(set(self.features)):
            raise ValueError("features must be unique")
        if self.target in self.features:
            raise ValueError("target cannot also be a feature")
        if self.column_roles.get(self.target) != ColumnRole.TARGET:
            raise ValueError("target must have the target column role")
        if sum(role == ColumnRole.TARGET for role in self.column_roles.values()) != 1:
            raise ValueError("exactly one target column role is required")
        invalid_features = [
            feature
            for feature in self.features
            if self.column_roles.get(feature) != ColumnRole.FEATURE
        ]
        if invalid_features:
            raise ValueError("every feature must have the feature column role")
        if len({model.id for model in self.models}) != len(self.models):
            raise ValueError("model ids must be unique")
        if len({step.id for step in self.preprocessing}) != len(self.preprocessing):
            raise ValueError("preprocessing step ids must be unique")
        if not any(model.enabled for model in self.models):
            raise ValueError("at least one model must be enabled")
        metric_ids = [self.metrics.primary, *self.metrics.secondary]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric ids must be unique")
        if self.task == TaskType.MULTICLASS_CLASSIFICATION:
            if any(metric in {"precision", "recall", "f1"} for metric in metric_ids):
                if self.metrics.averaging is None:
                    raise ValueError("multiclass precision, recall, and f1 require averaging")
            if "roc_auc" in metric_ids and self.metrics.roc_auc_strategy is None:
                raise ValueError("multiclass roc_auc requires an explicit strategy")
        if self.task == TaskType.BINARY_CLASSIFICATION:
            positive_metrics = {"precision", "recall", "f1", "roc_auc", "pr_auc"}
            if positive_metrics.intersection(metric_ids) and self.metrics.positive_label is None:
                raise ValueError("binary threshold and ranking metrics require positive_label")
        if self.tuning.enabled and self.tuning.metric is None:
            raise ValueError("enabled tuning requires an optimization metric")
        return self
