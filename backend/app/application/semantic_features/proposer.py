import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.application.datasets.service import DatasetService
from app.application.semantic_features.columns import semantic_column_options
from app.domain.decision.semantic_features import (
    SemanticDiscoveryMode,
    SemanticQuestionProposal,
)
from app.domain.exceptions import DecisionBenchmarkError
from app.llm.base import LLMProvider, LLMRequest
from app.llm.errors import LLMResponseError


class _DraftQuestion(BaseModel):
    """Esquema intermediário de rascunho de questão semântica gerada pelo LLM."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input_columns: list[str]
    type: Literal["noul", "choice", "score"]
    instructions: str
    options: list[str]


class _DraftProposal(BaseModel):
    """Proposta preliminar de questões semânticas extraída da resposta do provedor."""

    model_config = ConfigDict(extra="forbid")

    questions: list[_DraftQuestion]


def _validate_proposal(draft: _DraftProposal, language: str) -> SemanticQuestionProposal:
    """Valida e converte o rascunho de proposta do LLM na entidade de domínio estrita.

    Args:
        draft (_DraftProposal): Rascunho estruturado retornado pelo LLM.
        language (str): Idioma das perguntas ('pt-BR' ou 'en-US').

    Returns:
        SemanticQuestionProposal: Proposta canônica validada pelo domínio.

    Raises:
        LLMResponseError: Se as opções forem duplicadas, inválidas ou violarem os schemas.
    """
    questions = []
    for item in draft.questions:
        if item.type == "noul":
            if len(item.options) not in (0, 2):
                raise LLMResponseError("LLM provider returned invalid semantic question options.")
            criteria = {"true": item.options[0], "false": item.options[1]} if item.options else None
        else:
            if len(item.options) != len({option.strip() for option in item.options}):
                raise LLMResponseError("LLM provider returned duplicate semantic question options.")
            if item.type == "choice":
                criteria = {option: None for option in item.options}
            else:
                criteria = item.options
        questions.append(
            {
                "id": item.id,
                "input_columns": item.input_columns,
                "question": {
                    "type": item.type,
                    "instructions": item.instructions,
                    "locale": language,
                    "criteria": criteria,
                },
            }
        )
    try:
        return SemanticQuestionProposal.model_validate({"questions": questions})
    except ValidationError as exc:
        raise LLMResponseError("LLM provider returned an invalid semantic proposal.") from exc


class SemanticProposalResult(BaseModel):
    """Resultado da descoberta de features semânticas contendo metadados de execução.

    Attributes:
        proposal (SemanticQuestionProposal): Proposta de questões geradas.
        provider_id (str): Identificador do modelo/provedor LLM utilizado.
        discovery_mode (SemanticDiscoveryMode): Modo de descoberta adotado.
        sample_sent (bool): Indica se uma amostra real do dataset foi enviada no prompt.
        sample_rows (int): Quantidade de linhas enviadas na amostra, se aplicável.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal: SemanticQuestionProposal
    provider_id: str
    discovery_mode: SemanticDiscoveryMode
    sample_sent: bool
    sample_rows: int


class SemanticFeatureProposer:
    """Propositor de novas features semânticas baseadas em colunas textuais via LLM.

    Attributes:
        datasets (DatasetService): Serviço para recuperação de perfis e amostras de dados.
        provider (LLMProvider): Provedor LLM configurado para inferência de sugestões.
        prompt_directory (Path): Diretório base de templates de prompt.
        send_sample (bool): Flag que controla envio de linhas de exemplo ao LLM.
        sample_size (int): Quantidade máxima de linhas enviadas na amostra.
        random_seed (int): Semente para determinismo nas gerações.
    """

    def __init__(
        self,
        *,
        datasets: DatasetService,
        provider: LLMProvider,
        prompt_directory: str | Path,
        send_sample: bool,
        sample_size: int,
        random_seed: int,
    ) -> None:
        """Inicializa o propositor de features semânticas com dependências e configurações.

        Args:
            datasets (DatasetService): Serviço de datasets.
            provider (LLMProvider): Provedor de IA generativa.
            prompt_directory (str | Path): Caminho dos diretórios de prompt.
            send_sample (bool): Se verdadeiro, envia amostras de texto no prompt.
            sample_size (int): Número de linhas de amostra a incluir.
            random_seed (int): Semente para reprodução das gerações.
        """
        self.datasets = datasets
        self.provider = provider
        self.prompt_directory = Path(prompt_directory)
        self.send_sample = send_sample
        self.sample_size = sample_size
        self.random_seed = random_seed

    def propose(
        self,
        *,
        dataset_id: str,
        objective: str,
        selected_columns: list[str],
        language: Literal["pt-BR", "en-US"],
        discovery_mode: SemanticDiscoveryMode,
    ) -> SemanticProposalResult:
        """Gera propostas de questões semânticas a partir das colunas textuais selecionadas.

        Args:
            dataset_id (str): Identificador do dataset.
            objective (str): Objetivo do modelo preditivo em linguagem natural.
            selected_columns (list[str]): Colunas textuais a analisar.
            language (Literal["pt-BR", "en-US"]): Idioma das instruções.
            discovery_mode (SemanticDiscoveryMode): Modo de descoberta semântica.

        Returns:
            SemanticProposalResult: Proposta estruturada contendo questões validadas.

        Raises:
            DecisionBenchmarkError: Se o modo for incompatível, colunas forem inválidas
                ou as questões geradas referenciem colunas não selecionadas.
        """
        if discovery_mode != SemanticDiscoveryMode.TARGET_AGNOSTIC:
            raise DecisionBenchmarkError(
                "Target-aware discovery requires an explicit development split and is not "
                "available through this endpoint"
            )
        profile = self.datasets.get_profile(dataset_id)
        if not selected_columns or len(selected_columns) != len(set(selected_columns)):
            raise DecisionBenchmarkError("Select at least one unique text column")
        unknown = set(selected_columns) - set(profile.column_names)
        if unknown:
            raise DecisionBenchmarkError(f"Unknown selected columns: {sorted(unknown)}")
        options = semantic_column_options(profile)
        unsupported = set(selected_columns) - set(options.recommended + options.optional)
        if unsupported:
            raise DecisionBenchmarkError(
                f"Semantic discovery requires eligible text columns: {sorted(unsupported)}"
            )
        profiles = [
            item.model_dump(mode="json")
            for item in profile.column_profiles
            if item.name in selected_columns
        ]
        payload: dict[str, object] = {
            "objective": objective,
            "language": language,
            "discovery_mode": discovery_mode.value,
            "selected_columns": selected_columns,
            "column_profiles": profiles,
        }
        sample_rows = 0
        if self.send_sample:
            sample = [
                {column: row.get(column) for column in selected_columns}
                for row in profile.preview[: self.sample_size]
            ]
            payload["untrusted_dataset_sample"] = sample
            sample_rows = len(sample)
        request = LLMRequest(
            system_prompt=(
                self.prompt_directory / language / "semantic_features_system.txt"
            ).read_text(encoding="utf-8"),
            user_prompt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            language=language,
            temperature=0,
            seed=self.random_seed,
        )
        draft = self.provider.generate_structured(request, _DraftProposal)
        proposal = _validate_proposal(draft, language)
        for question in proposal.questions:
            if not set(question.input_columns).issubset(selected_columns):
                raise DecisionBenchmarkError(
                    f"Proposed question {question.id} references an unselected column"
                )
        return SemanticProposalResult(
            proposal=proposal,
            provider_id=self.provider.id,
            discovery_mode=discovery_mode,
            sample_sent=self.send_sample,
            sample_rows=sample_rows,
        )
