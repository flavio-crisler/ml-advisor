from collections.abc import Iterable
from typing import Protocol

from app.domain.exceptions import DuplicateRegistryItemError, RegistryItemNotFoundError


class RegistryItem(Protocol):
    """Protocolo que define um item identificável capaz de ser registrado no Registry.

    Attributes:
        id (str): Identificador textual único do item.
    """

    id: str


class Registry[RegistryItemType: RegistryItem]:
    """Coleção tipada genérica para registro, recuperação e listagem de componentes por ID.

    Attributes:
        _items (dict[str, RegistryItemType]): Dicionário interno mapeando ID ao item registrado.
    """

    def __init__(self, items: Iterable[RegistryItemType] = ()) -> None:
        """Inicializa o registro populando com os itens informados.

        Args:
            items (Iterable[RegistryItemType]): Coleção inicial de itens a registrar.
        """
        self._items: dict[str, RegistryItemType] = {}
        for item in items:
            self.register(item)

    def register(self, item: RegistryItemType) -> None:
        """Registra um novo item no registro garantindo unicidade do ID.

        Args:
            item (RegistryItemType): Item a ser registrado.

        Raises:
            DuplicateRegistryItemError: Se o ID já estiver registrado previamente.
        """
        if item.id in self._items:
            raise DuplicateRegistryItemError(f"Registry id {item.id} is already registered")
        self._items[item.id] = item

    def get(self, item_id: str) -> RegistryItemType:
        """Recupera um item registrado pelo seu identificador.

        Args:
            item_id (str): Identificador único do item.

        Returns:
            RegistryItemType: Item registrado correspondente.

        Raises:
            RegistryItemNotFoundError: Se o ID não for encontrado no registro.
        """
        try:
            return self._items[item_id]
        except KeyError as exc:
            raise RegistryItemNotFoundError(f"Registry id {item_id} is not registered") from exc

    def contains(self, item_id: str) -> bool:
        """Verifica se um item com o ID especificado está presente no registro.

        Args:
            item_id (str): Identificador a checar.

        Returns:
            bool: True se o item estiver cadastrado, False caso contrário.
        """
        return item_id in self._items

    def all(self) -> tuple[RegistryItemType, ...]:
        """Retorna uma tupla contendo todos os itens atualmente registrados.

        Returns:
            tuple[RegistryItemType, ...]: Coleção de itens registrados.
        """
        return tuple(self._items.values())
