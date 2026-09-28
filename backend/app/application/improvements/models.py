from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.experiment.spec import (
    ExperimentSpec,
    MetricsSpec,
    ModelSpec,
    PreprocessingStepSpec,
    TuningSpec,
    ValidationSpec,
)

ImprovementLanguage = Literal["pt-BR", "en-US"]


class GeneratedSupervisedImprovementRecommendation(BaseModel):
    """Representa a sugestão de melhoria para experimento gerada pelo LLM.

    Attributes:
        summary (str): Resumo executivo da melhoria recomendada.
        rationale (list[str]): Justificativas técnicas para as mudanças propostas.
        models (list[ModelSpec]): Modelos de ML sugeridos para o novo experimento.
        preprocessing (list[PreprocessingStepSpec]): Etapas de pré-processamento sugeridas.
        validation (ValidationSpec): Estratégia de validação proposta.
        metrics (MetricsSpec): Especificação de métricas avaliativas.
        tuning (TuningSpec): Configurações de sintonia de hiperparâmetros.
        cautions (list[str]): Alertas sobre possíveis riscos ou trade-offs.
        unsupported_suggestions (list[str]): Sugestões conceituais fora do escopo automatizado.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=2000)
    rationale: list[str] = Field(default_factory=list, max_length=6)
    models: list[ModelSpec] = Field(min_length=1, max_length=6)
    preprocessing: list[PreprocessingStepSpec] = Field(default_factory=list, max_length=8)
    validation: ValidationSpec
    metrics: MetricsSpec
    tuning: TuningSpec = Field(default_factory=TuningSpec)
    cautions: list[str] = Field(default_factory=list, max_length=6)
    unsupported_suggestions: list[str] = Field(default_factory=list, max_length=6)


class SupervisedImprovementRecommendation(GeneratedSupervisedImprovementRecommendation):
    """Recomendação validada e enriquecida com metadados controlados pelo servidor.

    Attributes:
        schema_version (Literal["1.0"]): Versão do schema da recomendação.
        scope (Literal["supervised_experiment"]): Escopo da recomendação aplicada.
    """

    schema_version: Literal["1.0"] = "1.0"
    scope: Literal["supervised_experiment"] = "supervised_experiment"


class ConfigurationChange(BaseModel):
    """Descreve a alteração atômica em um campo de configuração entre especificações.

    Attributes:
        path (str): Caminho do campo modificado (ex: 'validation.strategy').
        before (str): Valor original formatado em string.
        after (str): Novo valor proposto formatado em string.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    before: str
    after: str


class SupervisedImprovementProposal(BaseModel):
    """Proposta completa de melhoria contendo diff, nova especificação e metadados.

    Attributes:
        schema_version (Literal["1.0"]): Versão do schema da proposta.
        source_experiment_id (str): ID do experimento de origem analisado.
        source_model_run_id (str | None): ID da execução específica de referência, se houver.
        source_spec_hash (str): Hash criptográfico da especificação de origem.
        recommendation (SupervisedImprovementRecommendation): Recomendação estruturada.
        proposed_spec (ExperimentSpec): Nova especificação executável completa do experimento.
        changes (list[ConfigurationChange]): Lista de mudanças pontuais entre especificações.
        provider_id (str): Identificador do provedor de LLM que gerou a proposta.
        is_local (bool): Indica se a proposta foi processada por modelo local.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    source_experiment_id: str
    source_model_run_id: str | None = None
    source_spec_hash: str
    recommendation: SupervisedImprovementRecommendation
    proposed_spec: ExperimentSpec
    changes: list[ConfigurationChange]
    provider_id: str
    is_local: bool


class ImprovementProposalRequest(BaseModel):
    """Parâmetros de requisição para solicitar proposta de melhoria a partir de experimento.

    Attributes:
        language (ImprovementLanguage): Idioma de preferência ('pt-BR' ou 'en-US').
        source_experiment_id (str): Identificador do experimento de origem.
        source_model_run_id (str | None): Identificador da execução específica, opcional.
        context (dict[str, Any]): Contexto adicional de métricas ou gráficos.
    """

    model_config = ConfigDict(extra="forbid")

    language: ImprovementLanguage
    source_experiment_id: str = Field(min_length=1)
    source_model_run_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ImprovementDraftRequest(BaseModel):
    """Dados de requisição para criar um experimento rascunho a partir de proposta aceita.

    Attributes:
        source_experiment_id (str): Identificador do experimento de origem.
        source_model_run_id (str | None): Identificador da execução específica de referência.
        source_spec_hash (str): Hash da especificação original para verificação de concorrência.
        provider_id (str): Identificador do provedor que gerou a recomendação.
        recommendation (SupervisedImprovementRecommendation): Recomendação a ser persistida.
    """

    model_config = ConfigDict(extra="forbid")

    source_experiment_id: str = Field(min_length=1)
    source_model_run_id: str | None = None
    source_spec_hash: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    recommendation: SupervisedImprovementRecommendation


class ImprovementDraftResult(BaseModel):
    """Resultado da criação de um novo experimento rascunho derivado de melhoria.

    Attributes:
        experiment_id (str): Identificador do novo experimento rascunho criado.
        parent_experiment_id (str): Identificador do experimento pai que serviu de base.
        name (str): Nome atribuído ao novo experimento.
        status (Literal["draft"]): Status inicial do experimento criado.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    parent_experiment_id: str
    name: str
    status: Literal["draft"] = "draft"
