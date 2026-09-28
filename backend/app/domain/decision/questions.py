import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _QuestionBase(BaseModel):
    """Classe base para especificações de perguntas direcionadas a provedores de decisão.

    Attributes:
        question_version (Literal["1.0"]): Versão do esquema estrutural da pergunta.
        instructions (str): Instruções detalhadas de negócio passadas ao provedor.
        locale (str): Código da localidade linguística adotada para a pergunta.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_version: Literal["1.0"] = "1.0"
    instructions: str = Field(min_length=1, max_length=4000)
    locale: str = Field(default="pt-BR", pattern=r"^[a-z]{2}(?:-[A-Z]{2})?$")

    @field_validator("instructions", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        """Remove espaços em branco supérfluos no início e fim das instruções.

        Args:
            value (object): Valor bruto de entrada.

        Returns:
            object: Texto limpo ou valor original.
        """
        return value.strip() if isinstance(value, str) else value


class NoulCriteria(BaseModel):
    """Define os critérios semânticos para decisões binárias ou afirmativas/negativas.

    Attributes:
        true (str | None): Descrição da condição para atribuição de valor verdadeiro.
        false (str | None): Descrição da condição para atribuição de valor falso.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    true: str | None = Field(default=None, min_length=1, max_length=4000)
    false: str | None = Field(default=None, min_length=1, max_length=4000)

    @field_validator("true", "false", mode="before")
    @classmethod
    def strip_descriptions(cls, value: object) -> object:
        """Remove espaços em branco ao redor dos critérios descritos.

        Args:
            value (object): Valor bruto do critério.

        Returns:
            object: Texto limpo ou valor original.
        """
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_description(self) -> "NoulCriteria":
        """Valida que pelo menos uma das condições (true ou false) esteja preenchida.

        Returns:
            NoulCriteria: Própria instância validada.

        Raises:
            ValueError: Se ambas as condições forem omitidas.
        """
        if self.true is None and self.false is None:
            raise ValueError("Noul criteria must describe true, false, or both")
        return self


class NoulQuestionSpec(_QuestionBase):
    """Especifica uma pergunta binária direcionada a um modelo de decisão.

    Attributes:
        type (Literal["noul"]): Identificador discriminador do tipo binário.
        criteria (NoulCriteria | None): Critérios descritivos opcionais de atribuição.
    """

    type: Literal["noul"] = "noul"
    criteria: NoulCriteria | None = None


class ChoiceQuestionSpec(_QuestionBase):
    """Especifica uma questão de escolha entre alternativas discretas categorizadas.

    Attributes:
        type (Literal["choice"]): Identificador discriminador de múltipla escolha.
        criteria (dict[str, str | None]): Dicionário com opções válidas e descrições.
    """

    type: Literal["choice"] = "choice"
    criteria: dict[str, str | None] = Field(min_length=2, max_length=255)

    @field_validator("criteria", mode="before")
    @classmethod
    def normalize_option_metadata(cls, value: object) -> object:
        """Normaliza metadados de opções fornecidos em formato condensado.

        Args:
            value (object): Estrutura de critérios antes da validação.

        Returns:
            object: Dicionário normalizado de opções.
        """
        if not isinstance(value, dict) or "options" not in value:
            return value
        raw_options = value.get("options")
        if not isinstance(raw_options, str):
            return value
        options = [option.strip() for option in re.split(r"[,;|]", raw_options) if option.strip()]
        if len(options) < 2 or len(options) != len(set(options)):
            return value
        return {option: None for option in options}

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, values: dict[str, str | None]) -> dict[str, str | None]:
        """Valida se as alternativas de escolha são não vazias e não reservadas.

        Args:
            values (dict[str, str | None]): Dicionário bruto de opções e descrições.

        Returns:
            dict[str, str | None]: Dicionário higienizado de alternativas.

        Raises:
            ValueError: Se houver opções em branco ou palavras-chave reservadas.
        """
        cleaned = {
            option.strip(): description.strip() if isinstance(description, str) else None
            for option, description in values.items()
        }
        if any(not option for option in cleaned):
            raise ValueError("choice option names must not be blank")
        if any(description == "" for description in cleaned.values()):
            raise ValueError("choice descriptions must be non-blank or null")
        reserved = {"options", "choices", "answer", "multi_select"}
        if set(cleaned).issubset(reserved):
            raise ValueError("choice criteria must use actual option names as keys")
        return cleaned


class ScoreQuestionSpec(_QuestionBase):
    """Especifica uma questão ordinal baseada em escala ou rubricas graduadas.

    Attributes:
        type (Literal["score"]): Identificador discriminador de pontuação/escala.
        criteria (list[str]): Lista ordenada com as faixas ou níveis de avaliação.
    """

    type: Literal["score"] = "score"
    criteria: list[str] = Field(min_length=2, max_length=10)

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, values: list[str]) -> list[str]:
        """Valida que todos os níveis da escala contenham descrições não vazias.

        Args:
            values (list[str]): Lista original de critérios de graduação.

        Returns:
            list[str]: Lista de critérios com espaços limpos.

        Raises:
            ValueError: Se algum critério estiver em branco.
        """
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("score criteria must not contain blank values")
        return cleaned


DecisionQuestionSpec = Annotated[
    NoulQuestionSpec | ChoiceQuestionSpec | ScoreQuestionSpec,
    Field(discriminator="type"),
]
