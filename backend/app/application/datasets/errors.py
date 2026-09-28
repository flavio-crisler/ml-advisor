class DatasetIngestionError(Exception):
    """Exceção base para falhas durante a ingestão e validação de datasets.

    Attributes:
        code (str): Código identificador do erro na API.
        message_key (str): Chave i18n para internacionalização da mensagem de erro.
    """

    code = "dataset_invalid"
    message_key = "datasets.errors.dataset_invalid"


class DatasetTooLargeError(DatasetIngestionError):
    """Exceção disparada quando o arquivo de dataset ultrapassa o limite máximo de tamanho.

    Attributes:
        code (str): Código identificador de dataset excedente.
        message_key (str): Chave i18n para a mensagem de erro.
    """

    code = "dataset_too_large"
    message_key = "datasets.errors.dataset_too_large"


class UnsupportedDatasetFormatError(DatasetIngestionError):
    """Exceção disparada quando o formato ou extensão do arquivo não é suportado.

    Attributes:
        code (str): Código identificador de formato não suportado.
        message_key (str): Chave i18n para a mensagem de erro.
    """

    code = "dataset_format_unsupported"
    message_key = "datasets.errors.dataset_format_unsupported"


class DatasetNotFoundError(DatasetIngestionError):
    """Exceção disparada quando o dataset solicitado não é encontrado no workspace.

    Attributes:
        code (str): Código identificador de dataset não encontrado.
        message_key (str): Chave i18n para a mensagem de erro.
    """

    code = "dataset_not_found"
    message_key = "datasets.errors.dataset_not_found"
