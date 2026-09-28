from pathlib import Path, PurePosixPath

from app.domain.exceptions import ArtifactAlreadyExistsError
from app.infrastructure.storage.base import ArtifactStorage


class LocalFilesystemStorage(ArtifactStorage):
    """Implementa armazenamento de artefatos no sistema de arquivos local.

    Attributes:
        root (Path): Diretório raiz absoluto onde os artefatos são gravados.
    """

    def __init__(self, root: str | Path) -> None:
        """Inicializa o armazenamento local garantindo a criação do diretório raiz.

        Args:
            root (str | Path): Caminho base para os artefatos.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, artifact_id: str, content: bytes) -> str:
        """Salva o conteúdo binário de um artefato no sistema de arquivos local.

        Args:
            artifact_id (str): Identificador relativo do artefato em formato POSIX.
            content (bytes): Conteúdo em bytes a ser gravado em disco.

        Returns:
            str: Caminho relativo do artefato em relação à raiz de armazenamento.

        Raises:
            ArtifactAlreadyExistsError: Se o arquivo já existir previamente.
            ValueError: Se o identificador for inválido ou tentar escapar do diretório raiz.
        """
        path = self._path_for(artifact_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as artifact_file:
                artifact_file.write(content)
        except FileExistsError as exc:
            raise ArtifactAlreadyExistsError(f"Artifact {artifact_id} already exists") from exc
        return path.relative_to(self.root).as_posix()

    def load(self, artifact_id: str) -> bytes:
        """Lê os bytes de um artefato armazenado localmente.

        Args:
            artifact_id (str): Identificador relativo do artefato.

        Returns:
            bytes: Conteúdo binário lido do arquivo.

        Raises:
            FileNotFoundError: Se o arquivo não for encontrado no disco.
            ValueError: Se o identificador for inválido ou tentar escapar da raiz.
        """
        return self._path_for(artifact_id).read_bytes()

    def delete(self, artifact_id: str) -> None:
        """Remove o arquivo de um artefato do armazenamento local.

        Args:
            artifact_id (str): Identificador relativo do artefato.

        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
            ValueError: Se o identificador for inválido ou tentar escapar da raiz.
        """
        self._path_for(artifact_id).unlink()

    def _path_for(self, artifact_id: str) -> Path:
        """Resolve e valida o caminho absoluto seguro para um identificador de artefato.

        Args:
            artifact_id (str): Identificador relativo em formato POSIX.

        Returns:
            Path: Caminho absoluto seguro dentro da raiz de armazenamento.

        Raises:
            ValueError: Se o identificador contiver separadores inválidos, caminhos absolutos,
                referências de escape '..' ou tentar escapar da raiz configurada.
        """
        if not artifact_id or "\\" in artifact_id:
            raise ValueError("Artifact identifier must be a non-empty POSIX path")
        relative = PurePosixPath(artifact_id)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise ValueError("Artifact identifier cannot escape the storage root")
        path = (self.root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact identifier cannot escape the storage root")
        return path
