import json
from pathlib import Path
from typing import Any

from app.application.advisor.validation import SAFE_CONTRACT_ISSUES
from app.application.datasets.profiling import DatasetProfile
from app.llm.base import LLMRequest
from app.ml.registries.catalog import RegistryCatalog

SUPPORTED_LANGUAGES = frozenset({"pt-BR", "en-US"})


class AdvisorPromptBuilder:
    """Construtor de prompts estruturados para o Advisor de Machine Learning.

    Attributes:
        prompt_directory (Path): Diretório raiz contendo os templates de prompt por idioma.
    """

    def __init__(self, prompt_directory: str | Path) -> None:
        """Inicializa o construtor com o diretório base de templates.

        Args:
            prompt_directory: Caminho para o diretório de prompts por idioma.
        """
        self.prompt_directory = Path(prompt_directory)

    def build(
        self,
        *,
        objective: str,
        profile: DatasetProfile,
        catalog: RegistryCatalog,
        language: str,
        send_sample: bool,
        sample_size: int,
        random_seed: int,
    ) -> LLMRequest:
        """Constrói a requisição para o LLM combinando o perfil do dataset e catálogo.

        Args:
            objective: Objetivo de negócio descrito pelo usuário.
            profile: Perfil estatístico detalhado do conjunto de dados.
            catalog: Catálogo de componentes e capacidades de ML suportados.
            language: Idioma da solicitação ('pt-BR' ou 'en-US').
            send_sample: Indica se uma amostra das primeiras linhas deve ser incluída.
            sample_size: Quantidade de linhas a incluir caso send_sample seja True.
            random_seed: Semente para reprodutibilidade da inferência do modelo.

        Returns:
            LLMRequest estruturada pronta para envio ao provedor LLM.
        """
        locale = language if language in SUPPORTED_LANGUAGES else "pt-BR"
        system_prompt = (self.prompt_directory / locale / "advisor_system.txt").read_text(
            encoding="utf-8"
        )
        profile_payload = profile.model_dump(mode="json", exclude={"preview"})
        payload: dict[str, Any] = {
            "objective": objective,
            "language": locale,
            "dataset_profile": profile_payload,
            "available_components": {
                "models": [item.model_dump(mode="json") for item in catalog.models.all()],
                "preprocessors": [
                    item.model_dump(mode="json") for item in catalog.preprocessors.all()
                ],
                "metrics": [item.model_dump(mode="json") for item in catalog.metrics.all()],
                "validation_strategies": [
                    item.model_dump(mode="json") for item in catalog.validators.all()
                ],
                "tuners": [item.model_dump(mode="json") for item in catalog.tuners.all()],
            },
        }
        if send_sample:
            payload["sample"] = profile.preview[:sample_size]
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            language=locale,
            temperature=0,
            seed=random_seed,
        )

    @staticmethod
    def with_validation_feedback(request: LLMRequest, issues: tuple[str, ...]) -> LLMRequest:
        """Enriquece a requisição original com feedback estruturado de erros para retry.

        Args:
            request: Requisição LLM original enviada anteriormente.
            issues: Lista de inconsistências identificadas na proposta retornada.

        Returns:
            Nova LLMRequest contendo o bloco de validation_feedback no payload.
        """
        # Never echo provider-supplied values as instructions on a retry.
        categories = list(
            dict.fromkeys(
                issue if issue in SAFE_CONTRACT_ISSUES else issue.split(":", 1)[0]
                for issue in issues
            )
        )[:10]
        feedback = {
            "validation_errors": categories,
            "instruction": (
                "A proposta anterior foi rejeitada. Corrija os erros e retorne uma nova "
                "proposta completa. Em preprocessing[].id use somente IDs de "
                "available_components.preprocessors e inclua cada ID no máximo uma vez; "
                "nomes de colunas pertencem a features ou target, nunca a preprocessing. "
                "A coluna definida como target nunca deve constar na lista features. "
                "Em classificação binária com roc_auc, f1, precision ou recall, informe "
                "obrigatoriamente positive_label com o valor da classe positiva."
                if request.language == "pt-BR"
                else "The previous proposal was rejected. Fix the errors and return a "
                "complete new proposal. In preprocessing[].id use only IDs from "
                "available_components.preprocessors and include each ID at most once; "
                "column names belong in features or target, never in preprocessing. "
                "The column defined as target must never appear in the features list. "
                "For binary classification with roc_auc, f1, precision, or recall, "
                "positive_label must be specified with the positive class value."
            ),
        }
        payload = json.loads(request.user_prompt)
        payload["validation_feedback"] = feedback
        return request.model_copy(
            update={"user_prompt": json.dumps(payload, ensure_ascii=False, sort_keys=True)}
        )
