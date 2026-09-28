from typing import Any

from app.application.explanations.models import (
    ExplanationAvailability,
    ExplanationLanguage,
    ExplanationResult,
    ExplanationScope,
    ResultExplanation,
)
from app.application.explanations.prompts import (
    ExplanationPromptBuilder,
    expected_chart_ids,
)
from app.llm.base import LLMProvider, UnavailableLLMProvider
from app.llm.errors import LLMResponseError


class ExplanationService:
    """Orquestra a geração de explicações em linguagem natural via modelos LLM.

    Attributes:
        provider (LLMProvider): Provedor de LLM utilizado para geração de texto estruturado.
        prompts (ExplanationPromptBuilder): Construtor de prompts e sanitizador de contexto.
        random_seed (int): Semente para garantia de determinismo nas gerações.
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        prompts: ExplanationPromptBuilder,
        random_seed: int,
    ) -> None:
        """Inicializa o serviço de explicações com provedor e construtor de prompts.

        Args:
            provider (LLMProvider): Instância do provedor de LLM configurado.
            prompts (ExplanationPromptBuilder): Instância do construtor de prompts.
            random_seed (int): Semente pseudoaleatória para reprodutibilidade.
        """
        self.provider = provider
        self.prompts = prompts
        self.random_seed = random_seed

    def availability(self) -> ExplanationAvailability:
        """Verifica a disponibilidade operacional do provedor LLM configurado.

        Returns:
            ExplanationAvailability: Status de disponibilidade e metadados do provedor.
        """
        return ExplanationAvailability(
            available=not isinstance(self.provider, UnavailableLLMProvider),
            provider_id=self.provider.id,
            is_local=self.provider.capabilities.is_local,
        )

    def explain(
        self,
        *,
        scope: ExplanationScope,
        language: ExplanationLanguage,
        context: dict[str, Any],
    ) -> ExplanationResult:
        """Gera a explicação estruturada para o escopo e contexto solicitados.

        Args:
            scope (ExplanationScope): Escopo temático da análise explicativa.
            language (ExplanationLanguage): Idioma de resposta ('pt-BR' ou 'en-US').
            context (dict[str, Any]): Dados estatísticos e gráficos a serem explicados.

        Returns:
            ExplanationResult: Resultado estruturado contendo textos e gráficos explicados.

        Raises:
            LLMResponseError: Se o provedor retornar gráficos fora do contrato solicitado.
        """
        chart_ids = expected_chart_ids(scope=scope, context=context)
        request = self.prompts.build(
            scope=scope,
            language=language,
            context=context,
            seed=self.random_seed,
        )
        explanation = self.provider.generate_structured(request, ResultExplanation)
        chart_explanations = {}
        for chart in explanation.chart_explanations:
            chart_explanations.setdefault(chart.chart_id, chart)
        if set(chart_explanations) != set(chart_ids):
            raise LLMResponseError(
                "LLM provider returned chart explanations outside the requested contract."
            )
        explanation = explanation.model_copy(
            update={"chart_explanations": [chart_explanations[chart_id] for chart_id in chart_ids]}
        )
        return ExplanationResult(
            explanation=explanation,
            provider_id=self.provider.id,
            is_local=self.provider.capabilities.is_local,
        )
