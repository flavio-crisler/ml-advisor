from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.domain.decision.questions import DecisionQuestionSpec
from app.domain.decision.responses import DecisionResponse
from app.domain.decision.specifications import DecisionState

ProviderHealthStatus = Literal["available", "unavailable", "unknown", "disabled"]


class DecisionProviderCapabilities(BaseModel):
    """Representa o conjunto de recursos e modalidades suportadas por um provedor de decisão.

    Attributes:
        schema_version (Literal["1.0"]): Versão estrutural do esquema de capacidades.
        supports_noul (bool): Indica se o provedor suporta decisão de abstenção ou saída nula.
        supports_choice (bool): Indica suporte a respostas estruturadas de escolha múltipla.
        supports_score (bool): Indica capacidade de retornar estimativas de probabilidade contínua.
        supports_batch (bool): Indica se o provedor aceita processamento paralelo em lote.
        supports_model_version_resolution (bool): Se o provedor relata versão exata do modelo.
        is_external (bool): Se a inferência depende de rede e serviço externo remoto.
        max_context_tokens (int | None): Limite superior de tokens aceito no contexto.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    supports_noul: bool
    supports_choice: bool
    supports_score: bool
    supports_batch: bool = False
    supports_model_version_resolution: bool = False
    is_external: bool
    max_context_tokens: int | None = Field(default=None, gt=0)


class ProviderHealth(BaseModel):
    """Descreve o estado de saúde operacional e latência de um provedor de decisão.

    Attributes:
        schema_version (Literal["1.0"]): Versão estrutural do relatório de saúde.
        status (ProviderHealthStatus): Situação de disponibilidade do provedor.
        message_key (str | None): Chave i18n correspondente à mensagem descritiva.
        latency_ms (float | None): Tempo de resposta do teste de conectividade em ms.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    status: ProviderHealthStatus
    message_key: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    latency_ms: float | None = Field(default=None, ge=0)


class DecisionProvider(Protocol):
    """Define o protocolo para integração e execução de provedores de inferência de decisão.

    Attributes:
        id (str): Identificador exclusivo do provedor no catálogo.
    """

    id: str

    async def evaluate(
        self,
        state: DecisionState,
        questions: dict[str, DecisionQuestionSpec],
    ) -> DecisionResponse:
        """Executa a avaliação das questões de decisão sobre o estado informado.

        Args:
            state (DecisionState): Contexto com os atributos de entrada da linha.
            questions (dict[str, DecisionQuestionSpec]): Questões analíticas a responder.

        Returns:
            DecisionResponse: Respostas e justificativas retornadas pelo provedor.
        """
        ...

    def get_capabilities(self) -> DecisionProviderCapabilities:
        """Retorna o conjunto de capacidades funcionais do provedor.

        Returns:
            DecisionProviderCapabilities: Capacidades e restrições suportadas.
        """
        ...

    async def health_check(self) -> ProviderHealth:
        """Verifica a conectividade e disponibilidade operacional do provedor.

        Returns:
            ProviderHealth: Relatório com status de saúde e latência observada.
        """
        ...
