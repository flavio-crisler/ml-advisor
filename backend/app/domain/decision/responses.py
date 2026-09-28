import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _validate_distribution(probabilities: dict[object, float]) -> None:
    """Valida se uma distribuição discreta de probabilidades soma um e possui valores finitos.

    Args:
        probabilities (dict[object, float]): Mapeamento de desfechos para suas probabilidades.

    Raises:
        ValueError: Se a distribuição for vazia, contiver valores não finitos ou soma != 1.
    """
    if not probabilities:
        raise ValueError("probability distribution must not be empty")
    if any(not math.isfinite(value) for value in probabilities.values()):
        raise ValueError("probabilities must be finite")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")


class NoulDecisionAnswer(BaseModel):
    """Representa a resposta probabilística a uma decisão binária de sim/não.

    Attributes:
        type (Literal["noul"]): Identificador discriminador do tipo binário.
        probability_yes (float): Probabilidade estimada para o desfecho positivo [0, 1].
        confidence (float): Distância em relação à incerteza máxima (|2*p - 1|).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["noul"] = "noul"
    probability_yes: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_confidence(self) -> "NoulDecisionAnswer":
        """Assegura a relação matemática entre a confiança e a probabilidade estimada.

        Returns:
            NoulDecisionAnswer: Própria instância validada.

        Raises:
            ValueError: Se o valor de confiança divergir de |2*p - 1|.
        """
        expected = abs((2 * self.probability_yes) - 1)
        if not math.isclose(self.confidence, expected, abs_tol=1e-6):
            raise ValueError("Noul confidence must equal its distance from an uncertain answer")
        return self


class ChoiceDecisionAnswer(BaseModel):
    """Representa a escolha categórica e sua respectiva distribuição de probabilidade.

    Attributes:
        type (Literal["choice"]): Identificador discriminador de múltipla escolha.
        selected_choice (str): Alternativa vencedora com maior probabilidade.
        choice_probabilities (dict[str, float]): Distribuição de probabilidade das opções.
        confidence (float): Nível de confiança equivalente à probabilidade da escolha.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["choice"] = "choice"
    selected_choice: str = Field(min_length=1)
    choice_probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ChoiceDecisionAnswer":
        """Valida se a opção selecionada existe na distribuição e tem a maior probabilidade.

        Returns:
            ChoiceDecisionAnswer: Própria instância validada.

        Raises:
            ValueError: Se a escolha não existir ou não tiver a maior probabilidade.
        """
        _validate_distribution(self.choice_probabilities)
        if self.selected_choice not in self.choice_probabilities:
            raise ValueError("selected choice must be present in the distribution")
        expected = max(self.choice_probabilities.values())
        if not math.isclose(
            self.choice_probabilities[self.selected_choice], expected, abs_tol=1e-6
        ):
            raise ValueError("selected choice must have the largest probability")
        return self


class ScoreDecisionAnswer(BaseModel):
    """Representa uma pontuação ordinal estimada e seu valor esperado ponderado.

    Attributes:
        type (Literal["score"]): Identificador discriminador de pontuação em escala.
        score (float): Valor esperado ponderado da pontuação resultante.
        score_probabilities (dict[int, float]): Distribuição de probabilidade por nível ordinal.
        legend (dict[int, str]): Descrição opcional de cada nível da escala avaliada.
        confidence (float): Grau de confiança atribuído à estimativa.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["score"] = "score"
    score: float
    score_probabilities: dict[int, float]
    legend: dict[int, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ScoreDecisionAnswer":
        """Valida se os níveis ordinais iniciam em zero e se a pontuação reflete o valor esperado.

        Returns:
            ScoreDecisionAnswer: Própria instância validada.

        Raises:
            ValueError: Se os níveis forem descontínuos ou o score divergir do valor esperado.
        """
        _validate_distribution(self.score_probabilities)
        levels = sorted(self.score_probabilities)
        if levels != list(range(len(levels))):
            raise ValueError("score probability levels must be contiguous and start at zero")
        if self.legend and set(self.legend) != set(self.score_probabilities):
            raise ValueError("score legend must describe every probability level")
        expected_score = sum(
            level * probability for level, probability in self.score_probabilities.items()
        )
        if not math.isclose(self.score, expected_score, abs_tol=1e-6):
            raise ValueError("score must equal the expected value of its distribution")
        return self


DecisionAnswer = Annotated[
    NoulDecisionAnswer | ChoiceDecisionAnswer | ScoreDecisionAnswer,
    Field(discriminator="type"),
]


class DecisionUsage(BaseModel):
    """Consolida as métricas de consumo de tokens, requisições e custos de inferência.

    Attributes:
        request_count (int): Total de requisições de rede efetuadas.
        input_tokens (int | None): Total de tokens do prompt de entrada.
        output_tokens (int | None): Total de tokens gerados na resposta.
        estimated_cost (float | None): Custo monetário estimado da operação em dólares.
        cache_hit (bool): Indica se o resultado foi recuperado de cache persistente.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int = Field(default=1, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    cache_hit: bool = False


class DecisionResponse(BaseModel):
    """Representa a resposta estruturada consolidada emitida por um provedor de decisão.

    Attributes:
        response_version (Literal["1.0"]): Versão do formato de resposta.
        provider_id (str): Identificador do provedor que produziu as respostas.
        requested_model (str): Nome do modelo de linguagem solicitado.
        resolved_model (str | None): Identificador exato da versão do modelo executado.
        answers (dict[str, DecisionAnswer]): Dicionário com respostas por ID de pergunta.
        usage (DecisionUsage): Informações de consumo de tokens e custos.
        latency_ms (float | None): Tempo total de processamento em milissegundos.
        request_id (str | None): Identificador único da requisição remota.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_version: Literal["1.0"] = "1.0"
    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    requested_model: str = Field(min_length=1)
    resolved_model: str | None = Field(default=None, min_length=1)
    answers: dict[str, DecisionAnswer] = Field(min_length=1)
    usage: DecisionUsage = Field(default_factory=DecisionUsage)
    latency_ms: float | None = Field(default=None, ge=0)
    request_id: str | None = Field(default=None, min_length=1)
