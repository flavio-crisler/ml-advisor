from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

ImbalanceStrategy = Literal["none", "class_weight"]


class ClassWeightResolver:
    """Calcula e resolve pesos balanceados para classes desbalanceadas.

    Attributes:
        strategy (ImbalanceStrategy): Estratégia de balanceamento ('none' ou 'class_weight').
        class_weights (dict[Any, float]): Dicionário associando cada rótulo de classe ao peso.
    """

    def __init__(self, strategy: ImbalanceStrategy) -> None:
        """Inicializa o resolvedor de pesos de classe.

        Args:
            strategy (ImbalanceStrategy): Estratégia de balanceamento.
        """
        self.strategy = strategy
        self.class_weights: dict[Any, float] = {}

    def fit(self, target: pd.Series) -> "ClassWeightResolver":
        """Calcula os pesos balanceados inversamente proporcionais às frequências das classes.

        Args:
            target (pd.Series): Série contendo os valores alvo do conjunto de treino.

        Returns:
            ClassWeightResolver: Instância ajustada do resolvedor.
        """
        self.class_weights = {}
        if self.strategy == "class_weight":
            classes = np.unique(target)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=target)
            self.class_weights = {
                label.item() if isinstance(label, np.generic) else label: float(weight)
                for label, weight in zip(classes, weights, strict=True)
            }
        return self

    def get_model_options(self) -> Mapping[str, Any]:
        """Retorna as opções a serem passadas para o estimador durante o fit().

        Returns:
            Mapping[str, Any]: Dicionário contendo class_weight se ativo.
        """
        return {"class_weight": self.class_weights} if self.class_weights else {}

    def get_config(self) -> Mapping[str, Any]:
        """Retorna a configuração e pesos calculados para persistência ou inspeção.

        Returns:
            Mapping[str, Any]: Dicionário contendo a estratégia e pesos por classe.
        """
        return {
            "id": "imbalance",
            "strategy": self.strategy,
            "class_weights": {str(label): weight for label, weight in self.class_weights.items()},
        }
