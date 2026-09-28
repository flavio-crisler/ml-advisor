import json
from pathlib import Path
from typing import Any

from app.application.improvements.models import ImprovementLanguage
from app.domain.experiment.spec import ExperimentSpec
from app.llm.base import LLMRequest

MAX_IMPROVEMENT_CONTEXT_CHARACTERS = 200_000


class ImprovementContextError(ValueError):
    """Exceção levantada quando o contexto da melhoria ultrapassa limites ou é inválido.

    Attributes:
        code (str): Código de erro padronizado para contexto inválido.
    """

    code = "improvement_context_invalid"


class ImprovementPromptBuilder:
    """Constrói prompts estruturados para sugestões de melhorias via modelos LLM.

    Attributes:
        prompt_directory (Path): Diretório base contendo os templates de prompt por idioma.
    """

    def __init__(self, prompt_directory: str | Path) -> None:
        """Inicializa o construtor com o diretório raiz de prompts.

        Args:
            prompt_directory (str | Path): Caminho para os templates de prompts.
        """
        self.prompt_directory = Path(prompt_directory)

    def build(
        self,
        *,
        language: ImprovementLanguage,
        source_spec: ExperimentSpec,
        source_metrics: dict[str, Any] | None,
        context: dict[str, Any],
        capabilities: dict[str, Any],
        seed: int,
    ) -> LLMRequest:
        """Monta o objeto LLMRequest com especificação original, contexto e capacidades.

        Args:
            language (ImprovementLanguage): Idioma de destino ('pt-BR' ou 'en-US').
            source_spec (ExperimentSpec): Especificação do experimento de referência.
            source_metrics (dict[str, Any] | None): Métricas da execução analisada.
            context (dict[str, Any]): Contexto agregado dos resultados obtidos.
            capabilities (dict[str, Any]): Catálogo de capacidades ML permitidas.
            seed (int): Semente pseudoaleatória para controle determinístico.

        Returns:
            LLMRequest: Objeto de requisição estruturada pronto para inferência.

        Raises:
            ImprovementContextError: Se o payload JSON ultrapassar o limite suportado.
        """
        payload = json.dumps(
            {
                "language": language,
                "source_experiment_spec": source_spec.model_dump(mode="json"),
                "source_model_metrics": source_metrics,
                "aggregate_result_context": context,
                "applicable_capabilities": capabilities,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(payload) > MAX_IMPROVEMENT_CONTEXT_CHARACTERS:
            raise ImprovementContextError(
                "The improvement context exceeds the supported size limit."
            )
        system_prompt = (
            self.prompt_directory / language / "improvement_proposal_system.txt"
        ).read_text(encoding="utf-8")
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=payload,
            language=language,
            temperature=0,
            seed=seed,
        )
