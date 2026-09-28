from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Classe base declarativa do SQLAlchemy para mapeamento objeto-relacional.
    """


class DatasetRecord(Base):
    """Representa o registro persistido de um dataset na tabela 'datasets'.

    Attributes:
        id (str): Identificador único universal do dataset.
        name (str): Nome amigável de identificação do dataset.
        original_filename (str): Nome do arquivo original importado.
        content_hash (str): Hash SHA-256 do arquivo para deduplicação.
        row_count (int): Total de linhas catalogadas.
        column_count (int): Total de colunas encontradas.
        created_at (datetime): Data e hora de importação em UTC.
        storage_path (str): Caminho físico no disco do dataset original.
        profile_path (str | None): Caminho do arquivo JSON de perfil estatístico.
    """

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column("hash", String, nullable=False, unique=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    profile_path: Mapped[str | None] = mapped_column(String)


class ExperimentRecord(Base):
    """Representa a entidade de experimento persistida na tabela 'experiments'.

    Attributes:
        id (str): Identificador único do experimento.
        name (str): Nome de exibição do experimento.
        description (str | None): Detalhamento opcional do experimento.
        mode (str): Modo de criação ('auto', 'assisted', 'expert').
        status (str): Situação do experimento (ex: 'draft', 'completed').
        dataset_id (str): Chave estrangeira referenciando o dataset utilizado.
        parent_experiment_id (str | None): Chave estrangeira do experimento clonado.
        objective (str): Descrição textual do objetivo preditivo.
        spec_version (str): Versão da especificação técnica.
        experiment_spec_json (str): Especificação serializada em JSON.
        created_at (datetime): Timestamp de criação em UTC.
        updated_at (datetime): Timestamp da última alteração em UTC.
        deleted_at (datetime | None): Timestamp de exclusão lógica ou None se ativo.
    """

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    parent_experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"))
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    spec_version: Mapped[str] = mapped_column(String, nullable=False)
    experiment_spec_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)


class ExperimentRunRecord(Base):
    """Representa uma execução de modelo persistida na tabela 'experiment_runs'.

    Attributes:
        id (str): Identificador único da execução.
        experiment_id (str): Chave estrangeira vinculada ao experimento pai.
        model_id (str): Identificador do modelo de machine learning avaliado.
        run_index (int): Índice sequencial de execução do modelo.
        status (str): Estado da execução (queued, running, completed, etc.).
        started_at (datetime | None): Data/hora de início do treinamento.
        finished_at (datetime | None): Data/hora de conclusão do treinamento.
        duration_seconds (float | None): Tempo total de execução em segundos.
        random_seed (int): Semente aleatória configurada.
        parameters_json (str): Hiperparâmetros serializados em JSON.
        metrics_json (str | None): Métricas computadas em JSON.
        artifact_path (str | None): Caminho em disco do modelo treinado.
        error_message (str | None): Descrição de erro em caso de falha.
        execution_id (str | None): Identificador da rodada de execução em lote.
    """

    __tablename__ = "experiment_runs"
    __table_args__ = (UniqueConstraint("experiment_id", "model_id", "run_index"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    run_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    artifact_path: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    execution_id: Mapped[str | None] = mapped_column(String, nullable=True)


class TuningTrialRecord(Base):
    """Representa uma tentativa de sintonia na tabela 'tuning_trials'.

    Attributes:
        id (str): Identificador único da tentativa.
        run_id (str): Chave estrangeira da execução de modelo vinculada.
        trial_number (int): Número sequencial da tentativa.
        parameters_json (str): Hiperparâmetros testados serializados em JSON.
        metric_value (float | None): Valor da métrica objetivo alcançado.
        status (str): Situação da tentativa (completed, pruned, failed).
    """

    __tablename__ = "tuning_trials"
    __table_args__ = (UniqueConstraint("run_id", "trial_number"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("experiment_runs.id"), nullable=False)
    trial_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String, nullable=False)


class SettingRecord(Base):
    """Representa uma chave-valor de configuração na tabela 'settings'.

    Attributes:
        key (str): Chave única identificadora do parâmetro.
        value (str): Valor textual armazenado.
        updated_at (datetime): Data e hora da última modificação em UTC.
    """

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ExperimentStrategyRecord(Base):
    """Representa uma estratégia gerada na tabela 'experiment_strategies'.

    Attributes:
        id (str): Identificador único da estratégia.
        experiment_id (str): Chave estrangeira exclusiva do experimento.
        source (str): Origem da estratégia ('auto' ou 'assisted').
        provider_id (str | None): Identificador do provedor de IA se assistido.
        strategy_json (str): Conteúdo estruturado da proposta em JSON.
        created_at (datetime): Timestamp de criação em UTC.
    """

    __tablename__ = "experiment_strategies"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id"), nullable=False, unique=True
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    provider_id: Mapped[str | None] = mapped_column(String)
    strategy_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class DecisionRunRecord(Base):
    """Representa uma execução de decisão zero-shot na tabela 'decision_runs'.

    Attributes:
        id (str): Identificador único da execução de benchmark.
        dataset_id (str): Chave estrangeira referenciando o dataset avaliado.
        experiment_id (str | None): Chave estrangeira do experimento se vinculado.
        provider_id (str): Identificador do provedor de inferência.
        requested_model (str): Nome do modelo originalmente solicitado.
        resolved_model (str | None): Versão exata do modelo retornado.
        decision_spec_version (str): Versão do contrato da especificação.
        question_spec_hash (str): Hash criptográfico da pergunta.
        spec_json (str): Especificação técnica serializada em JSON.
        status (str): Situação atual (queued, running, completed, etc.).
        row_count (int): Total de registros a serem processados.
        processed_rows (int): Linhas já inferidas com sucesso.
        request_count (int): Total de chamadas remotas de inferência.
        cache_hits (int): Quantidade de inferências obtidas via cache.
        input_tokens (int | None): Total acumulado de tokens de entrada.
        estimated_cost (float | None): Custo monetário acumulado em dólares.
        latency_ms_total (float | None): Tempo acumulado de resposta em milissegundos.
        artifact_path (str | None): Caminho do artefato de predições em disco.
        calibration_run_id (str | None): Identificador da calibração associada.
        error_message (str | None): Descrição de erro caso tenha falhado.
        created_at (datetime): Data e hora de criação em UTC.
        started_at (datetime | None): Início do processamento em UTC.
        finished_at (datetime | None): Conclusão do processamento em UTC.
    """

    __tablename__ = "decision_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"))
    provider_id: Mapped[str] = mapped_column(String, nullable=False)
    requested_model: Mapped[str] = mapped_column(String, nullable=False)
    resolved_model: Mapped[str | None] = mapped_column(String)
    decision_spec_version: Mapped[str] = mapped_column(String, nullable=False)
    question_spec_hash: Mapped[str] = mapped_column(String, nullable=False)
    spec_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost: Mapped[float | None] = mapped_column(Float)
    latency_ms_total: Mapped[float | None] = mapped_column(Float)
    artifact_path: Mapped[str | None] = mapped_column(String)
    calibration_run_id: Mapped[str | None] = mapped_column(String)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class CalibrationRunRecord(Base):
    """Representa uma calibração persistida na tabela 'calibration_runs'.

    Attributes:
        id (str): Identificador único da execução de calibração.
        decision_run_id (str): Chave estrangeira exclusiva da execução de decisão.
        task (str): Identificador da tarefa analítica (ex: binary, multiclass).
        metrics_json (str): Métricas de calibração serializadas em JSON.
        artifact_path (str): Caminho físico para gravação do relatório de calibração.
        created_at (datetime): Timestamp de criação em UTC.
    """

    __tablename__ = "calibration_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    decision_run_id: Mapped[str] = mapped_column(
        ForeignKey("decision_runs.id"), nullable=False, unique=True
    )
    task: Mapped[str] = mapped_column(String, nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class DecisionCacheRecord(Base):
    """Representa uma inferência em cache na tabela 'decision_cache'.

    Attributes:
        context_key (str): Chave primária contextual composta.
        cache_key (str): Chave canônica única para auditoria de deduplicação.
        dataset_hash (str): Hash do dataset de origem da linha.
        row_hash (str): Hash do conteúdo da linha avaliada.
        question_spec_hash (str): Hash da especificação da questão.
        provider_id (str): Identificador do provedor que avaliou.
        requested_model (str): Nome do modelo originalmente solicitado.
        resolved_model (str): Versão exata do modelo utilizada.
        response_json (str): Resposta da inferência serializada em JSON.
        created_at (datetime): Data e hora de inclusão em cache em UTC.
    """

    __tablename__ = "decision_cache"

    context_key: Mapped[str] = mapped_column(String, primary_key=True)
    cache_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    dataset_hash: Mapped[str] = mapped_column(String, nullable=False)
    row_hash: Mapped[str] = mapped_column(String, nullable=False)
    question_spec_hash: Mapped[str] = mapped_column(String, nullable=False)
    provider_id: Mapped[str] = mapped_column(String, nullable=False)
    requested_model: Mapped[str] = mapped_column(String, nullable=False)
    resolved_model: Mapped[str] = mapped_column(String, nullable=False)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SemanticFeatureSetRecord(Base):
    """Representa um conjunto de features semânticas na tabela 'semantic_feature_sets'.

    Attributes:
        id (str): Identificador único do conjunto de features semânticas.
        dataset_id (str): Chave estrangeira referenciando o dataset original.
        source_experiment_id (str | None): Chave estrangeira de experimento referenciado.
        name (str): Nome amigável de identificação do conjunto.
        version (str): Versão estrutural do conjunto.
        status (str): Situação do processamento (queued, running, completed, etc.).
        discovery_mode (str): Modo de extração ('target_agnostic' ou 'target_aware').
        questions_frozen (int): Flag indicando se perguntas estão congeladas (0 ou 1).
        question_spec_json (str): Perguntas serializadas em JSON.
        provider_id (str): Provedor de inferência utilizado.
        requested_model (str): Modelo solicitado para o processo.
        resolved_model (str | None): Versão do modelo retornada pelo provedor.
        row_count (int): Quantidade total de linhas a enriquecer.
        processed_rows (int): Linhas processadas com sucesso.
        request_count (int): Requisições remotas enviadas.
        cache_hits (int): Inferências reaproveitadas de cache.
        input_tokens (int | None): Tokens de entrada consumidos.
        estimated_cost (float | None): Custo monetário acumulado em dólares.
        latency_ms_total (float | None): Tempo total de processamento em ms.
        artifact_path (str | None): Caminho do arquivo contendo o dataset enriquecido.
        impact_json (str | None): Relatório de impacto preditivo em JSON.
        error_message (str | None): Descrição de erro caso tenha falhado.
        created_at (datetime): Timestamp de criação em UTC.
        started_at (datetime | None): Timestamp de início em UTC.
        finished_at (datetime | None): Timestamp de término em UTC.
    """

    __tablename__ = "semantic_feature_sets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    source_experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    discovery_mode: Mapped[str] = mapped_column(String, nullable=False)
    questions_frozen: Mapped[int] = mapped_column(Integer, nullable=False)
    question_spec_json: Mapped[str] = mapped_column(Text, nullable=False)
    provider_id: Mapped[str] = mapped_column(String, nullable=False)
    requested_model: Mapped[str] = mapped_column(String, nullable=False)
    resolved_model: Mapped[str | None] = mapped_column(String)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_cost: Mapped[float | None] = mapped_column(Float)
    latency_ms_total: Mapped[float | None] = mapped_column(Float)
    artifact_path: Mapped[str | None] = mapped_column(String)
    impact_json: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class CascadeRunRecord(Base):
    """Representa uma execução de política de cascata na tabela 'cascade_runs'.

    Attributes:
        id (str): Identificador único da execução da cascata.
        decision_run_id (str): Chave estrangeira da execução de decisão avaliada.
        policy_json (str): Parâmetros e limiares da política em JSON.
        metrics_json (str): Métricas de cobertura, acurácia e custo em JSON.
        artifact_path (str): Caminho em disco das predições roteadas.
        created_at (datetime): Data e hora de execução em UTC.
    """

    __tablename__ = "cascade_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    decision_run_id: Mapped[str] = mapped_column(ForeignKey("decision_runs.id"), nullable=False)
    policy_json: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

