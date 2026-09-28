from pydantic import BaseModel, ConfigDict, Field

from app.domain.experiment.spec import (
    ExperimentSpec,
    MetricsSpec,
    ModelSpec,
    PreprocessingStepSpec,
    TaskType,
    TuningSpec,
    ValidationSpec,
)


class RecommendationSpec(BaseModel):
    """Especificação estruturada de recomendação de modelagem gerada pelo Advisor.

    Attributes:
        task (TaskType): Tipo da tarefa de modelagem recomendada.
        target (str): Nome da coluna definida como variável alvo.
        features (list[str]): Lista de features selecionadas para o modelo.
        models (list[ModelSpec]): Modelos propostos para experimentação.
        preprocessing (list[PreprocessingStepSpec]): Etapas de pré-processamento sugeridas.
        validation (ValidationSpec): Estratégia de validação recomendada.
        metrics (MetricsSpec): Métricas primárias e secundárias recomendadas.
        tuning (TuningSpec): Configuração de busca e otimização de hiperparâmetros.
        explanation (str): Justificativa técnica para as escolhas propostas.
        assumptions (list[str]): Premissas adotadas durante a elaboração do plano.
        warnings (list[str]): Alertas ou observações críticas identificadas nos dados.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: TaskType
    target: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    models: list[ModelSpec] = Field(min_length=1)
    preprocessing: list[PreprocessingStepSpec] = Field(default_factory=list)
    validation: ValidationSpec
    metrics: MetricsSpec
    tuning: TuningSpec = Field(default_factory=TuningSpec)
    explanation: str = Field(min_length=1, max_length=4000)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    warnings: list[str] = Field(default_factory=list, max_length=20)


class AdvisorResult(BaseModel):
    """Resultado consolidado retornado pelo serviço Advisor de Machine Learning.

    Attributes:
        recommendation (RecommendationSpec): Recomendação técnica com explicações e alertas.
        proposal (ExperimentSpec): Especificação executável de experimento proposta.
        provider_id (str): Identificador do provedor LLM que gerou a resposta.
        sample_sent (bool): Indica se uma amostra de dados foi enviada ao provedor.
        sample_rows (int): Quantidade de linhas incluídas na amostra de dados enviada.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommendation: RecommendationSpec
    proposal: ExperimentSpec
    provider_id: str
    sample_sent: bool
    sample_rows: int
