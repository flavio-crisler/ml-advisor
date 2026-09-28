import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.application.datasets.errors import (
    DatasetIngestionError,
    DatasetNotFoundError,
    DatasetTooLargeError,
)
from app.application.datasets.importers import CsvDatasetImporter
from app.application.datasets.profiling import DatasetProfile, DeterministicDatasetProfiler
from app.domain.common import new_id
from app.domain.dataset.entities import Dataset
from app.domain.dataset.repositories import DatasetRepository
from app.infrastructure.storage.base import ArtifactStorage


@dataclass(frozen=True, slots=True)
class DatasetIngestionResult:
    """Resultado consolidado da operação de ingestão e perfilamento de um dataset.

    Attributes:
        dataset (Dataset): Entidade persistida correspondente ao dataset.
        profile (DatasetProfile): Perfil estatístico e semântico gerado.
        reused (bool): Indica se o dataset foi reutilizado a partir de hash pré-existente.
    """

    dataset: Dataset
    profile: DatasetProfile
    reused: bool


class DatasetService:
    """Serviço de aplicação responsável pelo ciclo de vida, upload e perfis de datasets.

    Attributes:
        repository (DatasetRepository): Repositório de persistência de entidades de datasets.
        storage (ArtifactStorage): Sistema de armazenamento para arquivos brutos e perfis.
        importer (CsvDatasetImporter): Importador determinístico de arquivos CSV.
        profiler (DeterministicDatasetProfiler): Gerador determinístico de perfis de dados.
        working_directory (Path): Diretório temporário para recepção de uploads em streaming.
        max_upload_bytes (int): Limite máximo em bytes permitido para o arquivo enviado.
    """

    _chunk_size = 1024 * 1024

    def __init__(
        self,
        *,
        repository: DatasetRepository,
        storage: ArtifactStorage,
        importer: CsvDatasetImporter,
        profiler: DeterministicDatasetProfiler,
        working_directory: str | Path,
        max_upload_bytes: int,
    ) -> None:
        """Inicializa o serviço de datasets com repositório, storage e importadores.

        Args:
            repository: Repositório para consulta e gravação de datasets.
            storage: Mecanismo de persistência de artefatos.
            importer: Componente de importação e validação de CSVs.
            profiler: Analisador determinístico para cálculo do perfil estatístico.
            working_directory: Caminho de trabalho para arquivos temporários.
            max_upload_bytes: Limite máximo configurado para tamanho de upload.
        """
        self.repository = repository
        self.storage = storage
        self.importer = importer
        self.profiler = profiler
        self.working_directory = Path(working_directory)
        self.max_upload_bytes = max_upload_bytes

    def ingest(
        self, *, filename: str, name: str | None, stream: BinaryIO
    ) -> DatasetIngestionResult:
        """Realiza a ingestão completa de um dataset via streaming binário.

        Valida o arquivo, calcula seu hash SHA-256 para deduplicação, executa a importação
        e o perfilamento determinístico e persiste o arquivo e o perfil no storage.

        Args:
            filename: Nome do arquivo enviado pelo cliente.
            name: Nome amigável opcional para o dataset.
            stream: Fluxo de bytes binários contendo os dados do arquivo.

        Returns:
            DatasetIngestionResult contendo o dataset, perfil e indicação se foi deduplicado.

        Raises:
            DatasetTooLargeError: Se o arquivo exceder o limite de upload em bytes.
            DatasetIngestionError: Se o arquivo estiver vazio, corrompido ou sem colunas.
        """
        self.importer.validate_filename(filename)
        self.working_directory.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            temporary_path, content_hash, size_bytes = self._copy_upload(stream)
            existing = self.repository.find_by_hash(content_hash)
            if existing is not None:
                return DatasetIngestionResult(
                    dataset=existing,
                    profile=self.get_profile(existing.id),
                    reused=True,
                )

            imported = self.importer.import_file(temporary_path)
            dataset_id = new_id()
            profile = self.profiler.profile(
                imported.frame,
                size_bytes=size_bytes,
                encoding=imported.encoding,
                delimiter=imported.delimiter,
            )
            original_path = f"{dataset_id}/original.csv"
            profile_path = f"{dataset_id}/profile.json"
            self.storage.save(original_path, temporary_path.read_bytes())
            try:
                self.storage.save(
                    profile_path,
                    profile.model_dump_json(indent=2).encode("utf-8"),
                )
                display_name = (name or Path(filename).stem).strip()
                if not display_name:
                    raise DatasetIngestionError("Dataset name cannot be empty")
                dataset = Dataset(
                    id=dataset_id,
                    name=display_name,
                    original_filename=filename,
                    content_hash=content_hash,
                    row_count=profile.rows,
                    column_count=profile.columns,
                    storage_path=original_path,
                    profile_path=profile_path,
                )
                self.repository.add(dataset)
            except Exception:
                self._delete_if_present(profile_path)
                self._delete_if_present(original_path)
                raise
            return DatasetIngestionResult(dataset=dataset, profile=profile, reused=False)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def list_datasets(self) -> list[Dataset]:
        """Lista todos os datasets registrados no workspace.

        Returns:
            Lista contendo todas as entidades de datasets cadastradas.
        """
        return self.repository.list_all()

    def get_dataset(self, dataset_id: str) -> Dataset:
        """Recupera a entidade de um dataset a partir de seu identificador único.

        Args:
            dataset_id: Identificador do dataset.

        Returns:
            Entidade Dataset correspondente.

        Raises:
            DatasetNotFoundError: Se o dataset não existir no repositório.
        """
        dataset = self.repository.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset {dataset_id} was not found")
        return dataset

    def get_profile(self, dataset_id: str) -> DatasetProfile:
        """Carrega e desserializa o perfil estatístico JSON persistido para o dataset.

        Args:
            dataset_id: Identificador do dataset associado ao perfil.

        Returns:
            Instância de DatasetProfile validada e carregada do storage.

        Raises:
            DatasetIngestionError: Se o perfil não estiver registrado para o dataset.
        """
        dataset = self.get_dataset(dataset_id)
        if dataset.profile_path is None:
            raise DatasetIngestionError("Dataset profile is unavailable")
        return DatasetProfile.model_validate_json(self.storage.load(dataset.profile_path))

    def _copy_upload(self, stream: BinaryIO) -> tuple[Path, str, int]:
        """Salva o stream do upload em arquivo temporário calculando hash e tamanho.

        Args:
            stream: Stream binário contendo os dados do arquivo enviado.

        Returns:
            Tupla com (caminho_temporario, hash_sha256, tamanho_em_bytes).

        Raises:
            DatasetTooLargeError: Se o tamanho exceder o limite máximo configurado.
            DatasetIngestionError: Se o arquivo estiver completamente vazio.
        """
        digest = hashlib.sha256()
        total = 0
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="upload-",
            suffix=".csv",
            dir=self.working_directory,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            while chunk := stream.read(self._chunk_size):
                total += len(chunk)
                if total > self.max_upload_bytes:
                    temporary_file.close()
                    temporary_path.unlink(missing_ok=True)
                    raise DatasetTooLargeError("Dataset exceeds configured upload limit")
                digest.update(chunk)
                temporary_file.write(chunk)
        if total == 0:
            temporary_path.unlink(missing_ok=True)
            raise DatasetIngestionError("Dataset file cannot be empty")
        return temporary_path, digest.hexdigest(), total

    def _delete_if_present(self, artifact_id: str) -> None:
        """Remove defensivamente um artefato do storage ignorando erros caso não exista.

        Args:
            artifact_id: Identificador do artefato a excluir.
        """
        try:
            self.storage.delete(artifact_id)
        except FileNotFoundError:
            pass
