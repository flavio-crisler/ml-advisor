from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.common import canonical_json, json_object, new_id, utc_now
from app.domain.decision.questions import DecisionQuestionSpec
from app.domain.decision.runs import DecisionRunStatus


class SemanticDiscoveryMode(StrEnum):
    """Representa a modalidade de descoberta e extração de variáveis semânticas."""

    TARGET_AGNOSTIC = "target_agnostic"
    TARGET_AWARE_DEVELOPMENT = "target_aware_development"


class SemanticQuestionSpec(BaseModel):
    """Define a especificação de uma pergunta para geração de feature semântica.

    Attributes:
        semantic_spec_version (Literal["1.0"]): Versão estrutural da especificação.
        id (str): Nome da nova coluna semântica a ser adicionada ao dataset.
        input_columns (list[str]): Colunas textuais ou categóricas usadas como contexto.
        question (DecisionQuestionSpec): Pergunta estruturada (noul, choice ou score).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_spec_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    input_columns: list[str] = Field(min_length=1, max_length=20)
    question: DecisionQuestionSpec = Field(discriminator="type")

    @field_validator("input_columns")
    @classmethod
    def validate_input_columns(cls, values: list[str]) -> list[str]:
        """Valida que as colunas de contexto sejam únicas e contenham nomes válidos.

        Args:
            values (list[str]): Nomes de colunas informados.

        Returns:
            list[str]: Lista de colunas limpas.

        Raises:
            ValueError: Se houver colunas em branco ou duplicadas.
        """
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("Semantic input columns must be unique and non-blank")
        return cleaned


class SemanticQuestionProposal(BaseModel):
    """Agrupa uma coleção de perguntas semânticas propostas para enriquecimento de dados.

    Attributes:
        schema_version (Literal["1.0"]): Versão do esquema estrutural da proposta.
        questions (list[SemanticQuestionSpec]): Coleção de especificações de perguntas semânticas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    questions: list[SemanticQuestionSpec] = Field(min_length=1, max_length=30)

    @field_validator("questions")
    @classmethod
    def unique_question_ids(cls, values: list[SemanticQuestionSpec]) -> list[SemanticQuestionSpec]:
        """Assegura a unicidade dos identificadores de colunas semânticas propostas.

        Args:
            values (list[SemanticQuestionSpec]): Lista de questões semânticas.

        Returns:
            list[SemanticQuestionSpec]: Própria lista se válida.

        Raises:
            ValueError: Se existirem identificadores duplicados.
        """
        if len({value.id for value in values}) != len(values):
            raise ValueError("Semantic question ids must be unique")
        return values


@dataclass(frozen=True, slots=True)
class SemanticFeatureSet:
    """Representa a execução em lote de extração de features semânticas e seu impacto.

    Attributes:
        dataset_id (str): Identificador do dataset de origem.
        name (str): Nome amigável de identificação do conjunto de features.
        version (str): Versão semântica do conjunto de features.
        discovery_mode (SemanticDiscoveryMode): Modo de descoberta adotado.
        questions_frozen (bool): Indica se a especificação de perguntas está congelada.
        question_spec_json (str): Perguntas serializadas em formato JSON canônico.
        provider_id (str): Provedor de IA executante das extrações.
        requested_model (str): Nome do modelo remoto solicitado.
        row_count (int): Quantidade total de linhas do dataset a transformar.
        source_experiment_id (str | None): Experimento de referência para cálculo de impacto.
        status (DecisionRunStatus): Situação de execução do processamento.
        resolved_model (str | None): Versão do modelo retornada pelo provedor.
        processed_rows (int): Quantidade de linhas processadas com sucesso.
        request_count (int): Total de requisições enviadas ao provedor.
        cache_hits (int): Quantidade de inferências obtidas via cache.
        input_tokens (int | None): Total acumulado de tokens de entrada.
        estimated_cost (float | None): Custo monetário acumulado em dólares.
        latency_ms_total (float | None): Tempo total de resposta em milissegundos.
        artifact_path (str | None): Caminho do novo dataset com as colunas semânticas.
        impact_json (str | None): Relatório de impacto comparativo serializado em JSON.
        error_message (str | None): Mensagem detalhada em caso de falha.
        id (str): Identificador único universal do conjunto de features.
        created_at (datetime): Data de criação do registro em UTC.
        started_at (datetime | None): Data de início do processamento.
        finished_at (datetime | None): Data de término do processamento.
    """

    dataset_id: str
    name: str
    version: str
    discovery_mode: SemanticDiscoveryMode
    questions_frozen: bool
    question_spec_json: str
    provider_id: str
    requested_model: str
    row_count: int
    source_experiment_id: str | None = None
    status: DecisionRunStatus = DecisionRunStatus.QUEUED
    resolved_model: str | None = None
    processed_rows: int = 0
    request_count: int = 0
    cache_hits: int = 0
    input_tokens: int | None = None
    estimated_cost: float | None = None
    latency_ms_total: float | None = None
    artifact_path: str | None = None
    impact_json: str | None = None
    error_message: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        name: str,
        discovery_mode: SemanticDiscoveryMode,
        questions: list[SemanticQuestionSpec],
        provider_id: str,
        requested_model: str,
        row_count: int,
        source_experiment_id: str | None = None,
    ) -> "SemanticFeatureSet":
        """Instancia um novo conjunto de features semânticas serializando a proposta de questões.

        Args:
            dataset_id (str): Identificador do dataset de origem.
            name (str): Nome descritivo do conjunto.
            discovery_mode (SemanticDiscoveryMode): Modo de descoberta utilizado.
            questions (list[SemanticQuestionSpec]): Lista de questões estruturadas.
            provider_id (str): Identificador do provedor de IA.
            requested_model (str): Nome do modelo solicitado.
            row_count (int): Total de registros a serem processados.
            source_experiment_id (str | None): Experimento base opcional para medir impacto.

        Returns:
            SemanticFeatureSet: Nova instância em fila para processamento.
        """
        proposal = SemanticQuestionProposal(questions=questions)
        return cls(
            dataset_id=dataset_id,
            source_experiment_id=source_experiment_id,
            name=name,
            version="1.0",
            discovery_mode=discovery_mode,
            questions_frozen=True,
            question_spec_json=canonical_json(proposal.model_dump(mode="json")),
            provider_id=provider_id,
            requested_model=requested_model,
            row_count=row_count,
        )

    @property
    def questions(self) -> list[SemanticQuestionSpec]:
        """Desserializa e retorna as questões semânticas associadas ao conjunto.

        Returns:
            list[SemanticQuestionSpec]: Lista de questões estruturadas.
        """
        payload = json_object(self.question_spec_json)
        return [SemanticQuestionSpec.model_validate(item) for item in payload["questions"]]

    @property
    def impact(self) -> dict[str, Any] | None:
        """Desserializa e retorna a análise de impacto preditivo nas métricas se disponível.

        Returns:
            dict[str, Any] | None: Relatório de impacto ou None se não avaliado.
        """
        return json_object(self.impact_json) if self.impact_json else None

    @staticmethod
    def serialize_impact(impact: Mapping[str, Any]) -> str:
        """Serializa o relatório de impacto preditivo em formato canônico.

        Args:
            impact (Mapping[str, Any]): Dicionário com as métricas de impacto comparadas.

        Returns:
            str: String JSON canônica.
        """
        return canonical_json(impact)
