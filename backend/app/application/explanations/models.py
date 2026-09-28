from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExplanationScope = Literal[
    "model_run",
    "decision_calibration",
    "confidence_cascade",
    "chart_explanation",
    "semantic_impact",
    "experiment_comparison",
]
ExplanationLanguage = Literal["pt-BR", "en-US"]


class ChartExplanation(BaseModel):
    """Representa a explicação interpretável de um gráfico gerado pelo modelo.

    Attributes:
        chart_id (str): Identificador do gráfico analisado.
        summary (str): Resumo textual da interpretação do gráfico.
        findings (list[str]): Principais achados ou padrões observados.
        cautions (list[str]): Alertas ou cuidados na interpretação das informações.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chart_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_-]+$")
    summary: str = Field(min_length=1, max_length=1200)
    findings: list[str] = Field(default_factory=list, max_length=4)
    cautions: list[str] = Field(default_factory=list, max_length=3)


class ResultExplanation(BaseModel):
    """Estrutura a explicação abrangente dos resultados de um modelo ou experimento.

    Attributes:
        summary (str): Resumo geral da análise e desempenho observado.
        findings (list[str]): Lista dos principais achados estatísticos e operacionais.
        cautions (list[str]): Alertas e pontos de atenção sobre os resultados.
        next_steps (list[str]): Sugestões de próximos passos recomendados.
        chart_explanations (list[ChartExplanation]): Lista de explicações de gráficos individuais.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1, max_length=2000)
    findings: list[str] = Field(default_factory=list, max_length=6)
    cautions: list[str] = Field(default_factory=list, max_length=6)
    next_steps: list[str] = Field(default_factory=list, max_length=6)
    chart_explanations: list[ChartExplanation] = Field(default_factory=list, max_length=20)


class ExplanationResult(BaseModel):
    """Encapsula o resultado de explicação gerada juntamente com metadados do provedor.

    Attributes:
        explanation (ResultExplanation): Conteúdo estruturado da explicação.
        provider_id (str): Identificador do provedor de LLM utilizado.
        is_local (bool): Indica se o modelo/provedor foi executado localmente.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    explanation: ResultExplanation
    provider_id: str
    is_local: bool


class ExplanationAvailability(BaseModel):
    """Indica a disponibilidade do serviço de explicações e o provedor configurado.

    Attributes:
        available (bool): Verdadeiro se o serviço de explicação está disponível.
        provider_id (str): Identificador do provedor padrão ou ativo.
        is_local (bool): Verdadeiro se o provedor opera localmente.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    provider_id: str
    is_local: bool
