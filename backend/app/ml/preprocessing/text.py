from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.preprocessing.base import FeatureTransformer


class TextVectorizer(FeatureTransformer):
    """Vetoriza colunas de texto livre usando TF-IDF com vocabulário limitado.

    Ajusta um vocabulário exclusivo por coluna textual utilizando apenas dados de treino.

    Attributes:
        columns (tuple[str, ...]): Colunas de texto a serem vetorizadas.
        max_features (int): Quantidade máxima de features/tokens por coluna.
        vectorizers (dict[str, TfidfVectorizer]): Instâncias de TfidfVectorizer
            ajustadas por coluna.
        output_columns (dict[str, tuple[str, ...]]): Mapeamento de coluna para nomes de
            features geradas.
    """

    def __init__(self, columns: list[str], *, max_features: int = 256) -> None:
        """Inicializa o vetorizador de texto.

        Args:
            columns (list[str]): Colunas textuais a vetorizar.
            max_features (int): Limite máximo de termos por coluna. Padrão 256.
        """
        self.columns = tuple(columns)
        self.max_features = max_features
        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.output_columns: dict[str, tuple[str, ...]] = {}

    def fit(self, features: pd.DataFrame, target: Any | None = None) -> "TextVectorizer":
        """Ajusta os estimadores TF-IDF nas colunas textuais presentes no DataFrame.

        Args:
            features (pd.DataFrame): DataFrame de treino.
            target (Any | None): Ignorado.

        Returns:
            TextVectorizer: Instância ajustada do vetorizador de texto.

        Raises:
            ValueError: Se uma coluna textual não contiver tokens utilizáveis.
        """
        del target
        self.vectorizers = {}
        self.output_columns = {}
        for column in self.columns:
            if column not in features:
                continue
            vectorizer = TfidfVectorizer(max_features=self.max_features)
            values = features[column].fillna("").astype(str)
            try:
                vectorizer.fit(values)
            except ValueError as exc:
                raise ValueError(
                    f"Text column {column} has no usable tokens in training rows"
                ) from exc
            self.vectorizers[column] = vectorizer
            self.output_columns[column] = tuple(
                f"tfidf__{column}__{token}" for token in vectorizer.get_feature_names_out()
            )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação TF-IDF nas colunas de texto e as substitui pelas matrizes.

        Args:
            features (pd.DataFrame): DataFrame com colunas textuais a transformar.

        Returns:
            pd.DataFrame: DataFrame com as colunas de texto substituídas pelas colunas TF-IDF.
        """
        if not self.vectorizers:
            return features.copy()
        transformed = features.drop(columns=list(self.vectorizers)).copy()
        for column, vectorizer in self.vectorizers.items():
            values = features[column].fillna("").astype(str)
            encoded = vectorizer.transform(values)
            frame = pd.DataFrame(
                encoded.toarray(), index=features.index, columns=self.output_columns[column]
            )
            transformed = pd.concat([transformed, frame], axis=1)
        return transformed

    def get_config(self) -> Mapping[str, Any]:
        """Retorna as configurações e colunas de saída geradas por cada coluna de texto.

        Returns:
            Mapping[str, Any]: Dicionário de configuração contendo vocabulários e nomes de features.
        """
        return {
            "id": "text_vectorization",
            "strategy": "tfidf",
            "kind": "feature",
            "input_columns": list(self.vectorizers),
            "max_features_per_column": self.max_features,
            "output_columns": {key: list(value) for key, value in self.output_columns.items()},
        }
