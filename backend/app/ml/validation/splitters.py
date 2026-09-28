from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from app.domain.exceptions import ValidationStrategyError
from app.domain.experiment.spec import TaskType, ValidationSpec


@dataclass(frozen=True, slots=True)
class ValidationFold:
    """Representa um fold de validação com os índices posicionais de treino e teste.

    Attributes:
        fold_index (int): Índice sequencial de identificação do fold.
        train_indices (tuple[int, ...]): Índices das amostras destinadas ao treinamento.
        test_indices (tuple[int, ...]): Índices das amostras destinadas à validação/teste.
    """

    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Converte o fold em um dicionário serializável.

        Returns:
            Dicionário com o índice do fold e as listas de índices de treino e teste.
        """
        return {
            "fold_index": self.fold_index,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
        }


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    """Plano consolidado de validação com estratégia e conjunto de folds.

    Attributes:
        strategy (str): Nome da estratégia de particionamento utilizada.
        random_seed (int): Semente utilizada na inicialização dos divisores.
        shuffle (bool): Indica se os dados foram embaralhados antes da divisão.
        folds (tuple[ValidationFold, ...]): Coleção imutável de folds gerados no particionamento.
    """

    strategy: str
    random_seed: int
    shuffle: bool
    folds: tuple[ValidationFold, ...]

    def to_dict(self) -> dict[str, Any]:
        """Converte a estrutura do plano de validação em um dicionário serializável.

        Returns:
            Dicionário com os parâmetros da estratégia e lista de folds.
        """
        return {
            "strategy": self.strategy,
            "random_seed": self.random_seed,
            "shuffle": self.shuffle,
            "folds": [fold.to_dict() for fold in self.folds],
        }


class ValidationSplitter:
    """Divisor e organizador de conjuntos de dados para treino e validação cruzada."""

    def split(
        self,
        specification: ValidationSpec,
        task: TaskType,
        target: Any,
    ) -> ValidationPlan:
        """Gera o plano de validação particionando os dados conforme a especificação.

        Args:
            specification: Especificação dos parâmetros de divisão e estratégia.
            task: Tipo da tarefa de machine learning (regressão, classificação).
            target: Vetor contendo a variável alvo (necessário para estratificação).

        Returns:
            ValidationPlan com todos os folds particionados.

        Raises:
            ValidationStrategyError: Se o vetor alvo for multidimensional, ou se a estratégia
                for incompatível com o tipo de tarefa.
        """
        target_values = np.asarray(target)
        if target_values.ndim != 1:
            raise ValidationStrategyError("Target must be one-dimensional")
        indices = np.arange(len(target_values))

        if specification.strategy == "train_test_split":
            stratify = (
                target_values if task != TaskType.REGRESSION and specification.shuffle else None
            )
            train, test = train_test_split(
                indices,
                test_size=specification.test_size,
                shuffle=specification.shuffle,
                random_state=specification.random_seed if specification.shuffle else None,
                stratify=stratify,
            )
            folds = (self._fold(0, train, test),)
        elif specification.strategy == "stratified_kfold":
            if task == TaskType.REGRESSION:
                raise ValidationStrategyError("stratified_kfold cannot be used for regression")
            splitter = StratifiedKFold(
                n_splits=specification.folds,
                shuffle=specification.shuffle,
                random_state=specification.random_seed if specification.shuffle else None,
            )
            folds = tuple(
                self._fold(index, train, test)
                for index, (train, test) in enumerate(splitter.split(indices, target_values))
            )
        elif specification.strategy == "kfold":
            if task != TaskType.REGRESSION:
                raise ValidationStrategyError(
                    "kfold is reserved for regression; use stratified_kfold for classification"
                )
            splitter = KFold(
                n_splits=specification.folds,
                shuffle=specification.shuffle,
                random_state=specification.random_seed if specification.shuffle else None,
            )
            folds = tuple(
                self._fold(index, train, test)
                for index, (train, test) in enumerate(splitter.split(indices))
            )
        else:
            raise ValidationStrategyError(f"Unknown validation strategy: {specification.strategy}")

        return ValidationPlan(
            strategy=specification.strategy,
            random_seed=specification.random_seed,
            shuffle=specification.shuffle,
            folds=folds,
        )

    @staticmethod
    def _fold(fold_index: int, train_indices: Any, test_indices: Any) -> ValidationFold:
        """Instancia um ValidationFold convertendo os índices para inteiros imutáveis.

        Args:
            fold_index: Índice sequencial do fold.
            train_indices: Coleção de índices das amostras de treino.
            test_indices: Coleção de índices das amostras de teste/validação.

        Returns:
            ValidationFold com os índices convertidos em tuplas de inteiros.
        """
        return ValidationFold(
            fold_index=fold_index,
            train_indices=tuple(int(index) for index in train_indices),
            test_indices=tuple(int(index) for index in test_indices),
        )
