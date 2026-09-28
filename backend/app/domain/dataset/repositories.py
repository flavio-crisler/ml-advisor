from typing import Protocol

from app.domain.dataset.entities import Dataset


class DatasetRepository(Protocol):
    """Define o contrato de persistência e consulta para entidades de Dataset."""

    def add(self, dataset: Dataset) -> Dataset:
        """Persiste um novo dataset no repositório.

        Args:
            dataset (Dataset): Entidade de dataset a ser persistida.

        Returns:
            Dataset: Instância do dataset salvo com identificadores atribuídos.
        """
        ...

    def get(self, dataset_id: str) -> Dataset | None:
        """Recupera um dataset pelo seu identificador único.

        Args:
            dataset_id (str): Identificador único universal do dataset.

        Returns:
            Dataset | None: Entidade encontrada ou None se não existir.
        """
        ...

    def find_by_hash(self, content_hash: str) -> Dataset | None:
        """Busca um dataset pelo hash criptográfico do seu conteúdo.

        Args:
            content_hash (str): Hash SHA-256 do arquivo original.

        Returns:
            Dataset | None: Entidade correspondente ou None se inexistente.
        """
        ...

    def list_all(self) -> list[Dataset]:
        """Lista todos os datasets persistidos ordenados por data de criação.

        Returns:
            list[Dataset]: Lista com todas as entidades de datasets registradas.
        """
        ...
