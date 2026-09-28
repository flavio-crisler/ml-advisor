import math
from dataclasses import dataclass
from typing import Any, Literal

from app.domain.exceptions import ExperimentSpecValidationError
from app.domain.experiment.spec import TaskType
from app.ml.registries.catalog import RegistryCatalog


@dataclass(frozen=True, slots=True)
class ParameterRule:
    """Define regras de tipo, limites e escolhas aceitas para um hiperparâmetro.

    Attributes:
        kind (Literal["integer", "number", "string", "scalar"]): Tipo esperado do valor.
        minimum (float | None): Valor numérico mínimo permitido, se aplicável.
        maximum (float | None): Valor numérico máximo permitido, se aplicável.
        choices (tuple[str | int | float | bool, ...]): Conjunto de valores discretos aceitos.
    """

    kind: Literal["integer", "number", "string", "scalar"]
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str | int | float | bool, ...] = ()

    def payload(self) -> dict[str, Any]:
        """Converte as restrições da regra em formato de dicionário para o catálogo.

        Returns:
            dict[str, Any]: Representação estruturada com tipo, limites e escolhas.
        """
        result: dict[str, Any] = {"type": self.kind}
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.choices:
            result["choices"] = list(self.choices)
        return result

    def accepts(self, value: Any) -> bool:
        """Verifica se um determinado valor cumpre as restrições definidas pela regra.

        Args:
            value (Any): Valor a ser testado.

        Returns:
            bool: Verdadeiro se o valor for compatível com as regras estabelecidas.
        """
        if self.kind == "integer":
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        elif self.kind == "number":
            valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif self.kind == "string":
            valid_type = isinstance(value, str)
        else:
            valid_type = value is None or isinstance(value, (str, int, float, bool))
        if not valid_type or (self.choices and value not in self.choices):
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                return False
            if self.minimum is not None and value < self.minimum:
                return False
            if self.maximum is not None and value > self.maximum:
                return False
        return True


BOOSTED = {
    "n_estimators": ParameterRule("integer", 50, 500),
    "max_depth": ParameterRule("integer", 2, 12),
    "learning_rate": ParameterRule("number", 0.01, 0.3),
    "subsample": ParameterRule("number", 0.5, 1.0),
    "colsample_bytree": ParameterRule("number", 0.5, 1.0),
}
RANDOM_FOREST = {
    "n_estimators": ParameterRule("integer", 50, 500),
    "max_depth": ParameterRule("integer", 2, 32),
    "min_samples_split": ParameterRule("integer", 2, 20),
}
CATBOOST = {
    "iterations": ParameterRule("integer", 50, 500),
    "depth": ParameterRule("integer", 3, 10),
    "learning_rate": ParameterRule("number", 0.01, 0.3),
    "l2_leaf_reg": ParameterRule("number", 1.0, 10.0),
}
LIGHTGBM = {
    "n_estimators": ParameterRule("integer", 50, 500),
    "num_leaves": ParameterRule("integer", 8, 128),
    "learning_rate": ParameterRule("number", 0.01, 0.3),
    "min_child_samples": ParameterRule("integer", 5, 100),
}
KNN = {
    "n_neighbors": ParameterRule("integer", 2, 50),
    "weights": ParameterRule("string", choices=("uniform", "distance")),
    "p": ParameterRule("integer", 1, 2),
}
SVM = {
    "C": ParameterRule("number", 0.001, 100.0),
    "kernel": ParameterRule("string", choices=("rbf", "linear", "poly")),
    "gamma": ParameterRule("string", choices=("scale", "auto")),
}

MODEL_PARAMETER_RULES: dict[str, dict[str, ParameterRule]] = {
    "logistic_regression": {"C": ParameterRule("number", 0.0001, 100.0)},
    "random_forest_classifier": RANDOM_FOREST,
    "random_forest_regressor": RANDOM_FOREST,
    "xgboost_classifier": BOOSTED,
    "xgboost_regressor": BOOSTED,
    "catboost_classifier": CATBOOST,
    "catboost_regressor": CATBOOST,
    "lightgbm_classifier": LIGHTGBM,
    "lightgbm_regressor": LIGHTGBM,
    "knn_classifier": KNN,
    "knn_regressor": KNN,
    "svm_classifier": SVM,
    "svr": {**SVM, "epsilon": ParameterRule("number", 0.001, 1.0)},
    "linear_regression": {},
}

PREPROCESSOR_PARAMETER_RULES: dict[str, dict[str, ParameterRule]] = {
    "text_vectorization": {"max_features": ParameterRule("integer", 1, 2048)},
    "missing_values": {"constant_value": ParameterRule("scalar")},
    "categorical_encoding": {},
    "boolean_encoding": {},
    "numeric_scaling": {},
    "datetime": {},
    "outliers": {
        "lower_quantile": ParameterRule("number", 0.0, 0.49),
        "upper_quantile": ParameterRule("number", 0.51, 1.0),
        "iqr_multiplier": ParameterRule("number", 0.1, 10.0),
    },
    "imbalance": {},
    "feature_selection": {"threshold": ParameterRule("number", 0.0, 1.0)},
}


class ImprovementCapabilityCatalog:
    """Catálogo de recursos e regras para sugestões de melhorias pelo módulo de IA.

    Attributes:
        catalog (RegistryCatalog): Catálogo central de modelos, pré-processadores e métricas.
    """

    def __init__(self, catalog: RegistryCatalog) -> None:
        """Inicializa o catálogo de capacidades com as definições de registros de ML.

        Args:
            catalog (RegistryCatalog): Catálogo contendo as definições de componentes ML.
        """
        self.catalog = catalog

    def for_task(self, task: TaskType) -> dict[str, Any]:
        """Gera o mapa de capacidades e parâmetros permitidos para a tarefa ML especificada.

        Args:
            task (TaskType): Tipo de tarefa supervisionada (classificação ou regressão).

        Returns:
            dict[str, Any]: Dicionário com modelos, pré-processamento, validação e métricas.
        """
        return {
            "models": [
                {
                    "id": item.id,
                    "parameters": {
                        key: rule.payload() for key, rule in MODEL_PARAMETER_RULES[item.id].items()
                    },
                    "capabilities": item.capabilities.model_dump(mode="json"),
                }
                for item in self.catalog.models.all()
                if task in item.tasks
            ],
            "preprocessing": [
                {
                    "id": item.id,
                    "strategies": list(item.strategies),
                    "parameters": {
                        key: rule.payload()
                        for key, rule in PREPROCESSOR_PARAMETER_RULES[item.id].items()
                    },
                }
                for item in self.catalog.preprocessors.all()
            ],
            "validation": [
                item.model_dump(mode="json")
                for item in self.catalog.validators.all()
                if task in item.tasks
            ],
            "metrics": [
                item.model_dump(mode="json")
                for item in self.catalog.metrics.all()
                if task in item.tasks
            ],
            "tuning": {
                "engines": [item.id for item in self.catalog.tuners.all()],
                "trials": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "protected_fields": [
                "dataset_id",
                "dataset_hash",
                "profile_version",
                "objective",
                "task",
                "target",
                "features",
                "column_roles",
            ],
        }

    def validate_parameters(
        self,
        *,
        task: TaskType,
        models: list[Any],
        preprocessing: list[Any],
    ) -> None:
        """Valida se os modelos e pré-processadores propostos respeitam as regras do catálogo.

        Args:
            task (TaskType): Tipo de tarefa de aprendizado supervisionado.
            models (list[Any]): Lista de especificações de modelos propostos.
            preprocessing (list[Any]): Lista de etapas de pré-processamento propostas.

        Raises:
            ExperimentSpecValidationError: Se houver modelos incompatíveis ou parâmetros inválidos.
        """
        issues: list[str] = []
        for model in models:
            rules = MODEL_PARAMETER_RULES.get(model.id)
            if rules is None or not self.catalog.models.contains(model.id):
                issues.append(f"ai_unknown_model:{model.id}")
                continue
            if task not in self.catalog.models.get(model.id).tasks:
                issues.append(f"ai_model_task_incompatible:{model.id}:{task.value}")
            issues.extend(self._parameter_issues(f"models.{model.id}", model.parameters, rules))
        for step in preprocessing:
            rules = PREPROCESSOR_PARAMETER_RULES.get(step.id)
            if rules is None:
                issues.append(f"ai_unknown_preprocessor:{step.id}")
                continue
            issues.extend(
                self._parameter_issues(f"preprocessing.{step.id}", step.parameters, rules)
            )
            if step.id == "outliers":
                lower = step.parameters.get("lower_quantile")
                upper = step.parameters.get("upper_quantile")
                if lower is not None and upper is not None and lower >= upper:
                    issues.append("ai_invalid_parameter:preprocessing.outliers:quantile_order")
        if issues:
            raise ExperimentSpecValidationError(issues)

    @staticmethod
    def _parameter_issues(
        prefix: str,
        values: dict[str, Any],
        rules: dict[str, ParameterRule],
    ) -> list[str]:
        """Verifica incoerências entre parâmetros fornecidos e regras estritas.

        Args:
            prefix (str): Prefixo de contexto para identificação do parâmetro no erro.
            values (dict[str, Any]): Dicionário de hiperparâmetros configurados.
            rules (dict[str, ParameterRule]): Dicionário com as regras para cada parâmetro.

        Returns:
            list[str]: Lista de strings com códigos de inconformidades encontradas.
        """
        issues: list[str] = []
        for key, value in values.items():
            rule = rules.get(key)
            if rule is None:
                issues.append(f"ai_unknown_parameter:{prefix}.{key}")
            elif not rule.accepts(value):
                issues.append(f"ai_invalid_parameter:{prefix}.{key}")
        return issues
