import type { components } from "../types/api.generated";

export type Experiment = components["schemas"]["ExperimentResponse"];
export type ExperimentSpec = components["schemas"]["ExperimentSpec"];
export type ModelRun = components["schemas"]["ModelRunResponse"];
export type TuningTrial = components["schemas"]["TuningTrialResponse"];
export type ComparisonRun = components["schemas"]["ComparisonRunResponse"];
export type Job = components["schemas"]["JobResponse"];
export type ExperimentComparison =
  components["schemas"]["ExperimentComparisonResponse"];
export type RankedModel = components["schemas"]["RankedModelResponse"];
export type ExcludedModel = components["schemas"]["ExcludedModelResponse"];
export type RoundSummary = components["schemas"]["RoundSummaryResponse"];
export type ComparisonProtocol =
  components["schemas"]["ComparisonProtocolResponse"];
export type ExperimentComparisonExplanationResult =
  components["schemas"]["ExplanationResult"];

export type PlotlyArtifact = {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
};

export type AnalysisArtifact = {
  schema_version: string;
  task: string;
  plots: Record<string, PlotlyArtifact>;
  explainability: {
    status: "available" | "unavailable";
    reason?: string;
    detail?: string;
  };
};

type ApiErrorPayload = {
  detail?: {
    code?: string;
    message?: string;
    message_key?: string;
    issues?: string[];
  };
};

/**
 * Erro customizado disparado em falhas de chamadas à API de experimentos.
 */
export class ExperimentApiError extends Error {
  /**
   * Inicializa uma nova instância de ExperimentApiError.
   *
   * @param code - Código único do erro da API.
   * @param messageKey - Chave i18n para exibição localizada de mensagens.
   * @param issues - Lista opcional de problemas estruturados ou falhas de validação.
   */
  constructor(
    public readonly code: string,
    public readonly messageKey?: string,
    public readonly issues: string[] = [],
  ) {
    super(code);
  }
}

/**
 * Trata respostas da API de experimentos desempacotando o JSON ou lançando erro estruturado.
 *
 * @param response - Objeto Response do fetch.
 * @returns Dados convertidos para o tipo genérico T.
 * @throws {ExperimentApiError} Se a resposta não for bem-sucedida.
 */
async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response
      .json()
      .catch(() => ({}))) as ApiErrorPayload;
    throw new ExperimentApiError(
      payload.detail?.code ?? "unknown",
      payload.detail?.message_key,
      payload.detail?.issues ?? [],
    );
  }
  return (await response.json()) as T;
}

/**
 * Lista todos os experimentos cadastrados no workspace.
 *
 * @returns Lista de experimentos com metadados e status de execução.
 */
export async function listExperiments(): Promise<Experiment[]> {
  return parseResponse<Experiment[]>(await fetch("/api/v1/experiments"));
}

/**
 * Cria um novo experimento com a especificação e modo selecionados.
 *
 * @param spec - Especificação técnica do experimento (dataset, target, modelos, etc.).
 * @param options - Opções adicionais como modo de execução e descrição opcional.
 * @returns Experimento criado e persistido.
 */
export async function createExperiment(
  spec: ExperimentSpec,
  options: { mode: "auto" | "assisted" | "expert"; description?: string },
): Promise<Experiment> {
  return parseResponse<Experiment>(
    await fetch("/api/v1/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...options, spec }),
    }),
  );
}

/**
 * Obtém os detalhes completos de um experimento específico pelo ID.
 *
 * @param experimentId - Identificador único do experimento.
 * @returns Experimento com especificação e histórico de execuções.
 */
export async function getExperiment(experimentId: string): Promise<Experiment> {
  return parseResponse<Experiment>(
    await fetch(`/api/v1/experiments/${experimentId}`),
  );
}

/**
 * Obtém os detalhes e métricas de uma execução individual de modelo (model run).
 *
 * @param runId - Identificador único da execução do modelo.
 * @returns Dados da execução incluindo métricas de treino/validação e artefatos.
 */
export async function getModelRun(runId: string): Promise<ModelRun> {
  return parseResponse<ModelRun>(
    await fetch(`/api/v1/experiments/runs/${runId}`),
  );
}

/**
 * Obtém os artefatos visuais e diagnósticos de análise de uma execução de modelo.
 *
 * @param runId - Identificador da execução do modelo.
 * @returns Artefatos com gráficos Plotly e explicabilidade SHAP.
 */
export async function getRunAnalysis(runId: string): Promise<AnalysisArtifact> {
  return parseResponse<AnalysisArtifact>(
    await fetch(`/api/v1/experiments/runs/${runId}/analysis`),
  );
}

/**
 * Obtém o histórico detalhado das tentativas (trials) do ajuste de hiperparâmetros de uma execução.
 *
 * @param runId - Identificador da execução do modelo.
 * @returns Lista de tentativas de tuning com parâmetros testados e pontuações obtidas.
 */
export async function getTuningTrials(runId: string): Promise<TuningTrial[]> {
  return parseResponse<TuningTrial[]>(
    await fetch(`/api/v1/experiments/runs/${runId}/tuning-trials`),
  );
}

/**
 * Clona um experimento existente com um novo nome.
 *
 * @param experimentId - Identificador do experimento de origem.
 * @param name - Novo nome para o experimento clonado.
 * @returns Experimento clonado recém-criado.
 */
export async function cloneExperiment(
  experimentId: string,
  name: string,
): Promise<Experiment> {
  return parseResponse<Experiment>(
    await fetch(`/api/v1/experiments/${experimentId}/clone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  );
}

/**
 * Atualiza a especificação de um experimento antes de sua execução.
 *
 * @param experimentId - Identificador do experimento a atualizar.
 * @param spec - Nova especificação a ser atribuída ao experimento.
 * @returns Experimento atualizado.
 */
export async function updateExperiment(
  experimentId: string,
  spec: ExperimentSpec,
): Promise<Experiment> {
  return parseResponse<Experiment>(
    await fetch(`/api/v1/experiments/${experimentId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }),
  );
}

/**
 * Renomeia um experimento existente.
 *
 * @param experimentId - Identificador do experimento a renomear.
 * @param name - Novo nome a ser atribuído.
 * @returns Experimento com o novo nome persistido.
 */
export async function renameExperiment(
  experimentId: string,
  name: string,
): Promise<Experiment> {
  return parseResponse<Experiment>(
    await fetch(`/api/v1/experiments/${experimentId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  );
}

/**
 * Exclui permanentemente um experimento e seus artefatos associados.
 *
 * @param experimentId - Identificador do experimento a excluir.
 */
export async function deleteExperiment(experimentId: string): Promise<void> {
  const response = await fetch(`/api/v1/experiments/${experimentId}`, {
    method: "DELETE",
  });
  if (!response.ok) await parseResponse<never>(response);
}

/**
 * Inicia a execução assíncrona do treinamento de um experimento.
 *
 * @param experimentId - Identificador do experimento a executar.
 * @returns Job assíncrono criado para o treinamento.
 */
export async function runExperiment(experimentId: string): Promise<Job> {
  return parseResponse<Job>(
    await fetch(`/api/v1/experiments/${experimentId}/run`, { method: "POST" }),
  );
}

/**
 * Cancela um treinamento de experimento atualmente em andamento.
 *
 * @param experimentId - Identificador do experimento a cancelar.
 * @returns Job cancelado com status atualizado.
 */
export async function cancelExperiment(experimentId: string): Promise<Job> {
  return parseResponse<Job>(
    await fetch(`/api/v1/experiments/${experimentId}/cancel`, {
      method: "POST",
    }),
  );
}

/**
 * Compara diretamente dois experimentos distintos.
 *
 * @param firstId - Identificador do primeiro experimento.
 * @param secondId - Identificador do segundo experimento.
 * @returns Lista de execuções comparativas pareadas.
 */
export async function compareExperiments(
  firstId: string,
  secondId: string,
): Promise<ComparisonRun[]> {
  return parseResponse<ComparisonRun[]>(
    await fetch(`/api/v1/experiments/${firstId}/compare/${secondId}`),
  );
}

/**
 * Recupera o relatório consolidado de comparação entre os modelos de uma execução do experimento.
 *
 * @param experimentId - Identificador do experimento.
 * @param executionId - Identificador opcional de uma execução específica.
 * @returns Comparação com modelos ranqueados, modelo campeão e protocolo.
 */
export async function getExperimentComparison(
  experimentId: string,
  executionId?: string,
): Promise<ExperimentComparison> {
  const query = executionId
    ? `?execution_id=${encodeURIComponent(executionId)}`
    : "";
  return parseResponse<ExperimentComparison>(
    await fetch(`/api/v1/experiments/${experimentId}/comparison${query}`),
  );
}

/**
 * Gera uma explicação em linguagem natural da comparação entre os modelos do experimento.
 *
 * @param experimentId - Identificador do experimento.
 * @param executionId - Identificador da execução do experimento.
 * @param language - Idioma desejado para a explicação gerada ("pt-BR" ou "en-US").
 * @returns Explicação gerada por IA com justificativas e métricas.
 */
export async function explainExperimentComparison(
  experimentId: string,
  executionId: string,
  language: "pt-BR" | "en-US",
): Promise<ExperimentComparisonExplanationResult> {
  return parseResponse<ExperimentComparisonExplanationResult>(
    await fetch(`/api/v1/experiments/${experimentId}/explanation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ execution_id: executionId, language }),
    }),
  );
}
