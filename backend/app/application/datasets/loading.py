import hashlib
import io

import pandas as pd

from app.application.datasets.errors import DatasetIngestionError
from app.application.datasets.profiling import DatasetProfile
from app.domain.dataset.entities import Dataset
from app.infrastructure.storage.base import ArtifactStorage


def load_verified_csv(
    dataset: Dataset,
    storage: ArtifactStorage,
) -> tuple[DatasetProfile, pd.DataFrame]:
    """Carrega e valida a integridade de um arquivo CSV persistido e seu perfil estatístico.

    Verifica se o hash SHA-256 do conteúdo carregado confere com o registrado na entidade do
    dataset, carrega o perfil estatístico imutável e reconstrói o DataFrame com o encoding
    e delimitador originais.

    Args:
        dataset: Entidade do dataset contendo o caminho do arquivo e hash de integridade.
        storage: Sistema de armazenamento de artefatos para leitura dos dados e perfil.

    Returns:
        Tupla contendo a instância de DatasetProfile e o DataFrame carregado.

    Raises:
        DatasetIngestionError: Se houver divergência no hash de integridade, ausência do
            perfil, falha no parse do CSV ou se o DataFrame resultante estiver vazio.
    """
    content = storage.load(dataset.storage_path)
    if hashlib.sha256(content).hexdigest() != dataset.content_hash:
        raise DatasetIngestionError("Persisted dataset content failed integrity verification")
    if dataset.profile_path is None:
        raise DatasetIngestionError("Dataset profile is unavailable")
    profile = DatasetProfile.model_validate_json(storage.load(dataset.profile_path))
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            encoding=profile.encoding,
            sep=profile.delimiter,
            low_memory=False,
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DatasetIngestionError("Persisted CSV could not be parsed") from error
    if frame.empty:
        raise DatasetIngestionError("Dataset must contain at least one data row")
    return profile, frame
