import json
from pathlib import Path
from typing import Any

from app.application.explanations.models import (
    ExplanationLanguage,
    ExplanationScope,
)
from app.llm.base import LLMRequest

MAX_EXPLANATION_CONTEXT_CHARACTERS = 200_000
MAX_EXPLANATION_CHARTS = 20


class ExplanationContextError(ValueError):
    """Exceção levantada quando o contexto fornecido para explicação é inválido.

    Attributes:
        code (str): Código identificador do erro de contexto.
    """

    code = "explanation_context_invalid"


def expected_chart_ids(*, scope: ExplanationScope, context: dict[str, Any]) -> tuple[str, ...]:
    """Extrai e valida os identificadores dos gráficos presentes no contexto da explicação.

    Args:
        scope (ExplanationScope): Escopo da explicação solicitada.
        context (dict[str, Any]): Dicionário com o contexto dos dados e gráficos.

    Returns:
        tuple[str, ...]: Tupla contendo os IDs validados e únicos dos gráficos.

    Raises:
        ExplanationContextError: Se os gráficos forem inválidos, duplicados ou se o escopo
            exigir exatamente um gráfico e receber quantidade diferente.
    """
    charts = context.get("charts")
    if charts is None:
        if scope == "chart_explanation":
            raise ExplanationContextError("A focused chart explanation requires exactly one chart.")
        return ()
    if not isinstance(charts, list) or len(charts) > MAX_EXPLANATION_CHARTS:
        raise ExplanationContextError("The chart context is invalid.")

    chart_ids: list[str] = []
    for chart in charts:
        if not isinstance(chart, dict):
            raise ExplanationContextError("The chart context is invalid.")
        chart_id = chart.get("chart_id")
        if (
            not isinstance(chart_id, str)
            or not chart_id
            or len(chart_id) > 100
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in chart_id
            )
        ):
            raise ExplanationContextError("The chart identifier is invalid.")
        chart_ids.append(chart_id)

    if len(chart_ids) != len(set(chart_ids)):
        raise ExplanationContextError("Chart identifiers must be unique.")
    if scope == "chart_explanation" and len(chart_ids) != 1:
        raise ExplanationContextError("A focused chart explanation requires exactly one chart.")
    return tuple(chart_ids)


class ExplanationPromptBuilder:
    """Constrói requisições estruturadas para o LLM a partir de prompts e contexto.

    Attributes:
        prompt_directory (Path): Diretório raiz onde os prompts localizados estão armazenados.
    """

    def __init__(self, prompt_directory: str | Path) -> None:
        """Inicializa o construtor com o caminho dos diretórios de prompts.

        Args:
            prompt_directory (str | Path): Caminho para o diretório de templates de prompt.
        """
        self.prompt_directory = Path(prompt_directory)

    def build(
        self,
        *,
        scope: ExplanationScope,
        language: ExplanationLanguage,
        context: dict[str, Any],
        seed: int,
    ) -> LLMRequest:
        """Monta o objeto LLMRequest serializando o contexto e carregando o system prompt.

        Args:
            scope (ExplanationScope): Escopo da explicação desejada.
            language (ExplanationLanguage): Idioma de destino ('pt-BR' ou 'en-US').
            context (dict[str, Any]): Dados e métricas de contexto a serem explicados.
            seed (int): Semente para determinismo na resposta do LLM.

        Returns:
            LLMRequest: Objeto pronto para envio ao provedor LLM.

        Raises:
            ExplanationContextError: Se o contexto serializado ultrapassar o limite de tamanho.
        """
        expected_chart_ids(scope=scope, context=context)
        serialized_context = json.dumps(
            {
                "scope": scope,
                "language": language,
                "context": context,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(serialized_context) > MAX_EXPLANATION_CONTEXT_CHARACTERS:
            raise ExplanationContextError(
                "The explanation context exceeds the supported size limit."
            )
        system_prompt = (
            self.prompt_directory / language / "result_explanation_system.txt"
        ).read_text(encoding="utf-8")
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=serialized_context,
            language=language,
            temperature=0,
            seed=seed,
        )
