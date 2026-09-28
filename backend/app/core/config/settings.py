from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Representa as configurações centralizadas da aplicação carregadas via ambiente.

    Attributes:
        app_env (str): Ambiente de execução da aplicação (ex: development, production).
        app_name (str): Nome amigável de exibição da aplicação.
        app_version (str): Versão semântica atual do sistema.
        log_level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]): Nível de logging.
        ui_language (str): Idioma padrão adotado na interface web.
        cors_origins_raw (str): String com origens autorizadas separadas por vírgula.
        database_url (str): String de conexão para o banco de dados relacional.
        artifact_storage (str): Mecanismo de persistência de arquivos e modelos.
        artifact_path (str): Diretório de gravação dos artefatos de experimentos.
        dataset_path (str): Diretório base de arquivos brutos de datasets.
        auth_mode (str): Modo de autenticação configurado na aplicação.
        max_upload_size_mb (int): Limite máximo em megabytes para upload de arquivos.
        dataset_preview_rows (int): Quantidade de linhas para exibição preliminar de datasets.
        llm_provider (Literal["ollama", "vllm", "openrouter"]): Provedor de LLM padrão.
        llm_model (str): Nome do modelo remoto padrão.
        llm_api_key (SecretStr): Chave de API secreta de autenticação com o provedor de IA.
        llm_base_url (str): Endereço base para chamadas de inferência de linguagem.
        reasoning_model (bool): Indica se o modelo selecionado suporta raciocínio explícito.
        llm_reasoning (str): Nível de esforço de raciocínio estruturado.
        llm_verbosity (str): Verbosidade do fluxo de pensamento do modelo.
        openrouter_provider (str): Lista de provedores de infraestrutura no OpenRouter.
        llm_language (str): Idioma utilizado na geração de recomendações e explicações.
        llm_send_sample (bool): Flag indicando se amostras de linhas devem acompanhar o prompt.
        llm_sample_size (int): Quantidade de linhas enviadas na amostragem.
        llm_timeout_seconds (float): Tempo limite em segundos para requisições de IA.
        advisor_prompts_path (str): Diretório local onde residem os templates de prompt.
        decision_features_enabled (bool): Flag para habilitação das rotas do Decision Lab.
        decision_provider (str): Provedor padrão para execução de inferências zero-shot.
        decision_model (str): Identificador do modelo para o Decision Lab.
        decision_base_url (str): URL base do provedor de decisão.
        decision_api_key (SecretStr): Chave de autenticação do provedor de decisão.
        decision_request_timeout_seconds (float): Tempo limite de inferência de decisão em segundos.
        decision_max_retries (int): Limite máximo de retentativas para chamadas transitórias.
        decision_concurrency (int): Quantidade de inferências simultâneas permitidas.
        decision_cache_enabled (bool): Indica se o cache determinístico está habilitado.
        decision_external_processing_ack_required (bool): Exige consentimento para envio externo.
        default_random_seed (int): Semente pseudoaleatória padrão adotada no sistema.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_name: str = "ML Advisor"
    app_version: str = "0.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    ui_language: str = "pt-BR"
    cors_origins_raw: str = "http://localhost:5173,http://localhost:8080"
    database_url: str = "sqlite:////data/database/app.db"
    artifact_storage: str = "local"
    artifact_path: str = "/data/experiments"
    dataset_path: str = "/data/datasets"
    auth_mode: str = "disabled"
    max_upload_size_mb: int = 50
    dataset_preview_rows: int = 20
    llm_provider: Literal["ollama", "vllm", "openrouter"] = "openrouter"
    llm_model: str = ""
    llm_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    llm_base_url: str = ""
    reasoning_model: bool = False
    llm_reasoning: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    llm_verbosity: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    openrouter_provider: str = ""
    llm_language: str = "pt-BR"
    llm_send_sample: bool = False
    llm_sample_size: int = Field(default=10, ge=1, le=100)
    llm_timeout_seconds: float = Field(default=60, gt=0, le=600)
    advisor_prompts_path: str = "/app/prompts"
    decision_features_enabled: bool = False
    decision_provider: str = "typesafe_jev"
    decision_model: str = ""
    decision_base_url: str = ""
    decision_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    decision_request_timeout_seconds: float = Field(default=60, gt=0, le=600)
    decision_max_retries: int = Field(default=3, ge=0, le=10)
    decision_concurrency: int = Field(default=8, ge=1, le=128)
    decision_cache_enabled: bool = True
    decision_external_processing_ack_required: bool = True
    default_random_seed: int = 41

    @property
    def cors_origins(self) -> list[str]:
        """Extrai e higieniza a lista de origens CORS autorizadas.

        Returns:
            list[str]: Lista de URLs autorizadas para comunicação cross-origin.
        """
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]

    @property
    def openrouter_providers(self) -> tuple[str, ...]:
        """Obtém a lista ordenada e sem duplicatas de provedores upstream do OpenRouter.

        Returns:
            tuple[str, ...]: Tupla de identificadores de provedores configurados.
        """
        return tuple(
            dict.fromkeys(
                provider.strip()
                for provider in self.openrouter_provider.split(",")
                if provider.strip()
            )
        )


@lru_cache
def get_settings() -> Settings:
    """Instancia e armazena em cache o singleton de configurações da aplicação.

    Returns:
        Settings: Instância com as configurações ativas carregadas do ambiente.
    """
    return Settings()
