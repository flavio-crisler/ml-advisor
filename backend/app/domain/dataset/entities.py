from dataclasses import dataclass, field
from datetime import datetime

from app.domain.common import new_id, utc_now


@dataclass(frozen=True, slots=True)
class Dataset:
    """Representa um conjunto de dados ingerido e catalogado na plataforma.

    Attributes:
        name (str): Nome amigável de exibição do dataset.
        original_filename (str): Nome do arquivo original submetido pelo usuário.
        content_hash (str): Hash SHA-256 para deduplicação de conteúdo.
        row_count (int): Quantidade de linhas tabulares registradas.
        column_count (int): Quantidade de colunas encontradas.
        storage_path (str): Caminho de armazenamento do arquivo original.
        profile_path (str | None): Caminho do arquivo contendo o perfil estruturado.
        id (str): Identificador único universal da entidade.
        created_at (datetime): Timestamp de criação do registro em UTC.
    """

    name: str
    original_filename: str
    content_hash: str
    row_count: int
    column_count: int
    storage_path: str
    profile_path: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Valida os invariantes de consistência e dimensões da entidade.

        Raises:
            ValueError: Se dados essenciais estiverem vazios ou dimensões forem negativas.
        """
        if not self.name.strip():
            raise ValueError("Dataset name cannot be empty")
        if not self.original_filename.strip():
            raise ValueError("Original filename cannot be empty")
        if not self.content_hash.strip():
            raise ValueError("Dataset hash cannot be empty")
        if self.row_count < 0 or self.column_count < 0:
            raise ValueError("Dataset dimensions cannot be negative")
