import re

from pydantic import BaseModel, ConfigDict

from app.application.datasets.profiling import ColumnType, DatasetProfile


class SemanticColumnOptions(BaseModel):
    """Opções de colunas textuais elegíveis para geração de features semânticas.

    Attributes:
        recommended (list[str]): Colunas recomendadas com texto predominantemente narrativo.
        optional (list[str]): Colunas opcionais com dados categóricos ou textuais estruturados.
        excluded_ids (list[str]): Colunas identificadas como IDs excluídas preventivamente.
    """

    model_config = ConfigDict(frozen=True)

    recommended: list[str]
    optional: list[str]
    excluded_ids: list[str]


def semantic_column_options(profile: DatasetProfile) -> SemanticColumnOptions:
    """Classifica colunas textuais do perfil em recomendadas, opcionais e identificadores.

    Args:
        profile (DatasetProfile): Perfil estrutural e estatístico do dataset analisado.

    Returns:
        SemanticColumnOptions: Conjunto de colunas categorizadas por prioridade semântica.
    """
    known_ids = set(profile.possible_id_columns)
    recommended: list[str] = []
    optional: list[str] = []
    excluded_ids: list[str] = []
    for column in profile.column_profiles:
        if column.inferred_type not in {ColumnType.TEXT, ColumnType.CATEGORICAL}:
            continue
        normalized_name = re.sub(r"[^a-z0-9]+", "_", column.name.casefold()).strip("_")
        is_id = (
            column.name in known_ids
            or normalized_name in {"id", "index"}
            or normalized_name.startswith("id_")
            or normalized_name.endswith("_id")
        )
        if is_id:
            excluded_ids.append(column.name)
        elif column.non_null_count and column.cardinality > 1:
            if column.inferred_type == ColumnType.TEXT:
                recommended.append(column.name)
            else:
                optional.append(column.name)
    return SemanticColumnOptions(
        recommended=recommended,
        optional=optional,
        excluded_ids=excluded_ids,
    )
