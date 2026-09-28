from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from app.domain.exceptions import ModelTaskIncompatibleError
from app.domain.experiment.spec import TaskType
from app.ml.models import adapters
from app.ml.models.base import ModelAdapter
from app.ml.registries.metadata import ModelMetadata
from app.ml.registries.registry import Registry

ModelAdapterFactory = Callable[[Mapping[str, object] | None], ModelAdapter]


@dataclass(frozen=True, slots=True)
class ModelAdapterRegistration:
    """Representa o registro associando metadados de um modelo à sua fábrica instanciadora.

    Attributes:
        metadata (ModelMetadata): Metadados declarativos do modelo (tarefas, requisitos).
        factory (ModelAdapterFactory): Função fábrica que instancia o adaptador do modelo.
    """

    metadata: ModelMetadata
    factory: ModelAdapterFactory

    @property
    def id(self) -> str:
        """Retorna o identificador textual único do modelo registrado.

        Returns:
            str: Identificador do modelo (ex: 'logistic_regression').
        """
        return self.metadata.id


class ModelAdapterRegistry:
    """Catálogo central de adaptadores de modelos registrados no sistema.

    Gerencia o ciclo de vida e instanciação de modelos garantindo compatibilidade
    com a tarefa do experimento.
    """

    def __init__(self, registrations: Iterable[ModelAdapterRegistration]) -> None:
        """Inicializa o catálogo com a coleção de registros fornecida.

        Args:
            registrations (Iterable[ModelAdapterRegistration]): Coleção de registros de modelos.
        """
        self._registrations = Registry(registrations)

    def create(
        self,
        model_id: str,
        task: TaskType,
        parameters: Mapping[str, object] | None = None,
    ) -> ModelAdapter:
        """Cria e retorna uma instância do adaptador do modelo solicitado.

        Args:
            model_id (str): Identificador do modelo desejado.
            task (TaskType): Tipo da tarefa do experimento (ex: BINARY, MULTICLASS, REGRESSION).
            parameters (Mapping[str, object] | None): Hiperparâmetros customizados opcionais.

        Returns:
            ModelAdapter: Instância configurada do adaptador de modelo.

        Raises:
            ModelTaskIncompatibleError: Se o modelo solicitado não der suporte à tarefa indicada.
        """
        registration = self._registrations.get(model_id)
        if task not in registration.metadata.tasks:
            raise ModelTaskIncompatibleError(f"Model {model_id} does not support task {task.value}")
        return registration.factory(parameters)

    def all(self) -> tuple[ModelAdapterRegistration, ...]:
        """Retorna todos os registros de adaptadores cadastrados no registro.

        Returns:
            tuple[ModelAdapterRegistration, ...]: Tupla contendo todos os registros.
        """
        return self._registrations.all()


def build_default_model_adapter_registry(
    metadata: Iterable[ModelMetadata],
) -> ModelAdapterRegistry:
    """Constrói o catálogo padrão de adaptadores de modelos mapeando metadados e fábricas.

    Args:
        metadata (Iterable[ModelMetadata]): Coleção de metadados de modelos cadastrados no catálogo.

    Returns:
        ModelAdapterRegistry: Registro pronto com todos os adaptadores de ML disponíveis.

    Raises:
        ValueError: Se houver divergência entre as fábricas implementadas e os metadados.
    """
    factories: dict[str, ModelAdapterFactory] = {
        "logistic_regression": adapters.logistic_regression,
        "random_forest_classifier": adapters.random_forest_classifier,
        "xgboost_classifier": adapters.xgboost_classifier,
        "catboost_classifier": adapters.catboost_classifier,
        "lightgbm_classifier": adapters.lightgbm_classifier,
        "knn_classifier": adapters.knn_classifier,
        "svm_classifier": adapters.svm_classifier,
        "linear_regression": adapters.linear_regression,
        "random_forest_regressor": adapters.random_forest_regressor,
        "xgboost_regressor": adapters.xgboost_regressor,
        "catboost_regressor": adapters.catboost_regressor,
        "lightgbm_regressor": adapters.lightgbm_regressor,
        "knn_regressor": adapters.knn_regressor,
        "svr": adapters.svr,
    }
    metadata_by_id = {item.id: item for item in metadata}
    if set(metadata_by_id) != set(factories):
        missing_factories = set(metadata_by_id) - set(factories)
        missing_metadata = set(factories) - set(metadata_by_id)
        raise ValueError(
            "Model registry mismatch: "
            f"missing factories={sorted(missing_factories)}, "
            f"missing metadata={sorted(missing_metadata)}"
        )
    return ModelAdapterRegistry(
        ModelAdapterRegistration(metadata=metadata_by_id[model_id], factory=factory)
        for model_id, factory in factories.items()
    )
