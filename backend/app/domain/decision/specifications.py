import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.decision.questions import DecisionQuestionSpec

DecisionScalar = str | int | float | bool | None


class DecisionState(BaseModel):
    """Representa os dados contextuais de uma linha específica sob avaliação de decisão.

    Attributes:
        row_id (str): Identificador exclusivo da linha ou registro no dataset.
        values (dict[str, DecisionScalar]): Mapeamento de nomes de colunas para valores escalares.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: str = Field(min_length=1, max_length=512)
    values: dict[str, DecisionScalar] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def validate_columns(cls, values: dict[str, DecisionScalar]) -> dict[str, DecisionScalar]:
        """Garante que todas as colunas de contexto possuam identificadores não vazios.

        Args:
            values (dict[str, DecisionScalar]): Dicionário de variáveis e valores.

        Returns:
            dict[str, DecisionScalar]: Dicionário validado de colunas.

        Raises:
            ValueError: Se houver nomes de colunas em branco.
        """
        if any(not column.strip() for column in values):
            raise ValueError("state column names must not be blank")
        return values


class DecisionSpec(BaseModel):
    """Especifica a configuração completa de inferência e questões para um provedor.

    Attributes:
        spec_version (Literal["1.0"]): Versão estrutural do esquema.
        provider (str): Identificador do provedor de decisão.
        model (str): Nome do modelo selecionado para avaliação.
        locale (str): Código de localidade linguística adotada.
        questions (dict[str, DecisionQuestionSpec]): Conjunto de perguntas estruturadas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: Literal["1.0"] = "1.0"
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    model: str = Field(min_length=1, max_length=256)
    locale: str = Field(default="pt-BR", pattern=r"^[a-z]{2}(?:-[A-Z]{2})?$")
    questions: dict[str, DecisionQuestionSpec] = Field(min_length=1, max_length=100)

    @field_validator("questions")
    @classmethod
    def validate_question_ids(
        cls, questions: dict[str, DecisionQuestionSpec]
    ) -> dict[str, DecisionQuestionSpec]:
        """Valida que os identificadores de questões atendam ao formato alfanumérico.

        Args:
            questions (dict[str, DecisionQuestionSpec]): Dicionário de questões por ID.

        Returns:
            dict[str, DecisionQuestionSpec]: Dicionário validado de questões.

        Raises:
            ValueError: Se o tamanho for inválido ou contiver caracteres especiais proibidos.
        """
        for question_id in questions:
            if not question_id or len(question_id) > 100:
                raise ValueError("question ids must contain between 1 and 100 characters")
            if not question_id.replace("_", "").replace("-", "").isalnum():
                raise ValueError(
                    "question ids may contain letters, numbers, hyphens, and underscores"
                )
        return questions

    def question_spec_hash(self) -> str:
        """Calcula o hash criptográfico SHA-256 da especificação canônica das perguntas.

        Returns:
            str: Hash hexadecimal representativo e determinístico do conjunto de perguntas.
        """
        payload = {
            "spec_version": self.spec_version,
            "locale": self.locale,
            "questions": {
                key: value.model_dump(mode="json") for key, value in sorted(self.questions.items())
            },
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
