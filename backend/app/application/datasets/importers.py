import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.application.datasets.errors import (
    DatasetIngestionError,
    UnsupportedDatasetFormatError,
)


@dataclass(frozen=True, slots=True)
class ImportedDataset:
    """Representa os dados carregados de um dataset e seus metadados de codificação.

    Attributes:
        frame (pd.DataFrame): DataFrame contendo os dados tabulares carregados.
        encoding (str): Codificação de caracteres detectada no arquivo.
        delimiter (str): Caractere delimitador de colunas identificado.
    """

    frame: pd.DataFrame
    encoding: str
    delimiter: str


class DatasetImporter(ABC):
    """Interface abstrata para importadores de arquivos de dados tabulares."""

    @abstractmethod
    def supports(self, filename: str) -> bool:
        """Verifica se o importador suporta o arquivo com base no nome e extensão.

        Args:
            filename: Nome do arquivo a ser verificado.

        Returns:
            True se o importador for capaz de processar o arquivo, False caso contrário.
        """

    @abstractmethod
    def import_file(self, path: Path) -> ImportedDataset:
        """Efetua o parse determinístico e carregamento do arquivo em um ImportedDataset.

        Args:
            path: Caminho do arquivo a ser importado.

        Returns:
            ImportedDataset contendo o DataFrame e metadados de codificação.
        """


class CsvDatasetImporter(DatasetImporter):
    """Importador determinístico de arquivos de dados delimitados por caracteres (CSV).

    Attributes:
        _delimiters (str): Conjunto de caracteres delimitadores suportados para detecção.
    """

    _delimiters = ",;\t|"

    def supports(self, filename: str) -> bool:
        """Verifica se o arquivo informado possui extensão .csv.

        Args:
            filename: Nome do arquivo a ser testado.

        Returns:
            True se a extensão for .csv (case-insensitive), False caso contrário.
        """
        return Path(filename).suffix.lower() == ".csv"

    def import_file(self, path: Path) -> ImportedDataset:
        """Lê e analisa o arquivo CSV determinando automaticamente codificação e delimitador.

        Args:
            path: Caminho no sistema de arquivos para o arquivo CSV.

        Returns:
            ImportedDataset contendo os dados em DataFrame e parâmetros de importação.

        Raises:
            DatasetIngestionError: Se o arquivo estiver corrompido, vazio ou sem colunas.
        """
        encoding, sample = self._decode_sample(path)
        delimiter = self._detect_delimiter(sample)
        try:
            frame = pd.read_csv(path, encoding=encoding, sep=delimiter, low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise DatasetIngestionError("CSV content could not be parsed") from exc
        if len(frame.columns) == 0:
            raise DatasetIngestionError("CSV must contain at least one column")
        return ImportedDataset(frame=frame, encoding=encoding, delimiter=delimiter)

    def validate_filename(self, filename: str) -> None:
        """Valida se o nome do arquivo é seguro e compatível com as regras de importação.

        Args:
            filename: Nome do arquivo enviado.

        Raises:
            UnsupportedDatasetFormatError: Se o nome contiver caminhos de diretório ou extensão
                diferente de .csv.
        """
        if not filename or Path(filename).name != filename:
            raise UnsupportedDatasetFormatError("Filename must not contain a path")
        if not self.supports(filename):
            raise UnsupportedDatasetFormatError("Only CSV files are supported")

    @staticmethod
    def _decode_sample(path: Path) -> tuple[str, str]:
        """Tenta decodificar uma amostra inicial do arquivo em UTF-8 com fallback para Latin-1.

        Args:
            path: Caminho do arquivo a ser amostrado.

        Returns:
            Tupla contendo a string de codificação e a amostra decodificada em texto.
        """
        sample = path.read_bytes()[:65_536]
        try:
            return "utf-8-sig", sample.decode("utf-8-sig")
        except UnicodeDecodeError:
            return "latin-1", sample.decode("latin-1")

    def _detect_delimiter(self, sample: str) -> str:
        """Detecta o delimitador de colunas a partir de uma amostra textual do arquivo.

        Args:
            sample: Amostra textual decodificada do início do arquivo.

        Returns:
            Caractere delimitador detectado ou vírgula como fallback seguro.
        """
        try:
            return csv.Sniffer().sniff(sample, delimiters=self._delimiters).delimiter
        except csv.Error:
            return ","
