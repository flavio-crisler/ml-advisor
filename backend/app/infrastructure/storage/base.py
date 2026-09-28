from abc import ABC, abstractmethod


class ArtifactStorage(ABC):
    """Define a interface abstrata para armazenamento de artefatos binários.

    Methods:
        save: Persiste bytes sem sobrescrever artefatos existentes.
        load: Carrega o conteúdo binário de um artefato existente.
        delete: Remove um artefato do armazenamento.
    """

    @abstractmethod
    def save(self, artifact_id: str, content: bytes) -> str:
        """Persiste bytes sob um identificador relativo sem sobrescrever artefatos.

        Args:
            artifact_id (str): Identificador relativo do artefato a persistir.
            content (bytes): Conteúdo binário a ser salvo.

        Returns:
            str: Caminho relativo do artefato persistido.

        Raises:
            ArtifactAlreadyExistsError: Se o identificador já existir.
        """

    @abstractmethod
    def load(self, artifact_id: str) -> bytes:
        """Carrega o conteúdo em bytes de um artefato pelo seu identificador.

        Args:
            artifact_id (str): Identificador relativo do artefato.

        Returns:
            bytes: Conteúdo binário carregado do armazenamento.

        Raises:
            FileNotFoundError: Se o artefato solicitado não for localizado.
        """

    @abstractmethod
    def delete(self, artifact_id: str) -> None:
        """Remove um artefato quando permitido pela política de ciclo de vida.

        Args:
            artifact_id (str): Identificador relativo do artefato a excluir.

        Raises:
            FileNotFoundError: Se o artefato a ser removido não existir.
        """
