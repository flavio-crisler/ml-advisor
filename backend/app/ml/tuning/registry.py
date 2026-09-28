from collections.abc import Callable, Iterable
from dataclasses import dataclass

from app.domain.exceptions import RegistryItemNotFoundError
from app.ml.tuning.base import TunerAdapter
from app.ml.tuning.optuna_adapter import OptunaTunerAdapter

TunerAdapterFactory = Callable[[], TunerAdapter]


@dataclass(frozen=True, slots=True)
class TunerAdapterRegistration:
    """Registro contendo o identificador e a fábrica de um adaptador de tuning.

    Attributes:
        id (str): Identificador único do adaptador de tuning.
        factory (TunerAdapterFactory): Função fábrica que instancia o TunerAdapter.
    """

    id: str
    factory: TunerAdapterFactory


class TunerAdapterRegistry:
    """Registro de fábricas de adaptadores de otimização de hiperparâmetros.

    Attributes:
        _registrations (dict[str, TunerAdapterRegistration]): Mapeamento interno dos
            adaptadores registrados.
    """

    def __init__(self, registrations: Iterable[TunerAdapterRegistration]) -> None:
        """Inicializa o registro com as fábricas de adaptadores de tuning.

        Args:
            registrations: Coleção iterável de registros de adaptadores.

        Raises:
            ValueError: Se houver identificadores duplicados nos registros fornecidos.
        """
        items = tuple(registrations)
        self._registrations = {item.id: item for item in items}
        if len(self._registrations) != len(items):
            raise ValueError("Tuner adapter ids must be unique")

    def create(self, tuner_id: str) -> TunerAdapter:
        """Instancia um novo adaptador de tuning com base em seu identificador.

        Args:
            tuner_id: Identificador único do adaptador requisitado.

        Returns:
            TunerAdapter: Instância criada do adaptador correspondente.

        Raises:
            RegistryItemNotFoundError: Se o identificador não existir no registro.
        """
        try:
            registration = self._registrations[tuner_id]
        except KeyError as error:
            raise RegistryItemNotFoundError(f"Unknown tuner adapter: {tuner_id}") from error
        return registration.factory()


def build_default_tuner_adapter_registry() -> TunerAdapterRegistry:
    """Cria e inicializa o registro padrão de adaptadores de tuning com Optuna pré-configurado.

    Returns:
        TunerAdapterRegistry: Instância do registro populada com o adaptador padrão 'optuna'.
    """
    return TunerAdapterRegistry([TunerAdapterRegistration(id="optuna", factory=OptunaTunerAdapter)])
