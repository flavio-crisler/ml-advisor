from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SemanticFeatureMetadata:
    """Metadados que descrevem as features semânticas adicionadas ao dataset.

    Attributes:
        feature_names (tuple[str, ...]): Nomes das colunas das features semânticas.
        column_types (dict[str, str]): Tipos inferidos/atribuídos para cada feature gerada.
    """

    feature_names: tuple[str, ...]
    column_types: dict[str, str]


class SemanticFeatureTransformer:
    """Alinha features semânticas persistidas em disco sem realizar requisições externas.

    Attributes:
        features (pd.DataFrame): DataFrame das features semânticas com coluna row_id.
        feature_names (tuple[str, ...]): Nomes das features semânticas a concatenar.
    """

    def __init__(self, features: pd.DataFrame) -> None:
        """Inicializa e valida a estrutura do DataFrame de features semânticas.

        Args:
            features (pd.DataFrame): Dados das features semânticas contendo row_id.

        Raises:
            ValueError: Se row_id estiver ausente, duplicado ou não houver features.
        """
        if "row_id" not in features:
            raise ValueError("Semantic feature artifact requires row_id")
        if features["row_id"].duplicated().any():
            raise ValueError("Semantic row ids must be unique")
        self.features = features.copy()
        self.features["row_id"] = self.features["row_id"].astype(str)
        self.feature_names = tuple(column for column in features if column != "row_id")
        if not self.feature_names:
            raise ValueError("Semantic feature artifact has no feature columns")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Mescla as features semânticas no DataFrame original baseado no índice row_id.

        Args:
            frame (pd.DataFrame): DataFrame de entrada do dataset.

        Returns:
            pd.DataFrame: DataFrame contendo as features originais e as semânticas acopladas.

        Raises:
            ValueError: Se houver linhas sem features semânticas ou conflito de nomes de colunas.
        """
        row_ids = pd.Series(frame.index.astype(str), index=frame.index, name="row_id")
        aligned = self.features.set_index("row_id").reindex(row_ids.tolist())
        if aligned.isna().all(axis=1).any():
            missing = row_ids[aligned.isna().all(axis=1).to_numpy()].tolist()
            raise ValueError(f"Semantic features are missing row ids: {missing[:5]}")
        aligned.index = frame.index
        overlap = set(frame.columns).intersection(self.feature_names)
        if overlap:
            raise ValueError(f"Semantic features collide with source columns: {sorted(overlap)}")
        return pd.concat([frame.copy(), aligned.loc[:, self.feature_names]], axis=1)

    def get_metadata(self) -> SemanticFeatureMetadata:
        """Retorna os metadados das features semânticas acopladas.

        Returns:
            SemanticFeatureMetadata: Definição de nomes de features e tipos.
        """
        return SemanticFeatureMetadata(
            feature_names=self.feature_names,
            column_types={feature: "numeric" for feature in self.feature_names},
        )
