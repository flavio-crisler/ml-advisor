type PlotArtifact = {
  data: Plotly.Data[];
};

/**
 * Contexto estruturado de gráfico agregado para consumo pelo serviço de explicações LLM.
 */
export type ExplanationChartContext = {
  chart_id: string;
  title: string;
  purpose: string;
  axes: { x: string; y: string };
  data_policy: "aggregate_only_no_raw_rows";
  data: Record<string, unknown>;
};

/**
 * Resumo estatístico e distribuição em bins de uma variável numérica.
 */
export type NumericDistribution = {
  count: number;
  statistics: Record<string, number> | null;
  bins: Array<{ lower: number; upper: number; count: number }>;
};

type ChartMetadata = Pick<
  ExplanationChartContext,
  "title" | "purpose" | "axes"
>;

type TuningTrialContext = {
  status: string;
  metric_value?: number | null;
  parameters?: Record<string, unknown>;
};

const MAX_SHAP_FEATURES_PER_TRACE = 20;

const EXACT_AGGREGATE_PLOTS = new Set([
  "confusion_matrix",
  "roc_curve",
  "precision_recall_curve",
  "feature_importance",
  "learning_curve",
]);

const HISTOGRAM_PLOTS = new Set([
  "probability_distribution",
  "residual_distribution",
]);

const PAIRED_SUMMARY_PLOTS = new Set(["actual_vs_predicted", "residual_plot"]);

/**
 * Constrói a lista de contextos de gráficos agregados para envio ao modelo de linguagem.
 *
 * @param plots - Mapa de artefatos de plotagem Plotly indexados pelo nome do gráfico.
 * @param metadata - Função que retorna título, propósito e eixos para um dado gráfico.
 * @returns Lista de contextos de gráficos prontos para compor o prompt explicativo.
 */
export function buildPlotExplanationCharts(
  plots: Record<string, PlotArtifact>,
  metadata: (plotName: string) => ChartMetadata,
): ExplanationChartContext[] {
  return Object.entries(plots).map(([plotName, plot]) => ({
    chart_id: plotName,
    ...metadata(plotName),
    data_policy: "aggregate_only_no_raw_rows",
    data: {
      traces: plot.data.map((trace) => summarizeTrace(plotName, trace)),
    },
  }));
}

/**
 * Resume artefatos de plotagem brutos em estruturas agregadas seguras para envio a LLMs.
 *
 * @param plots - Mapa de artefatos de plotagem Plotly brutos.
 * @returns Objeto com os traços sumarizados de cada gráfico.
 */
export function summarizePlotArtifacts(plots: Record<string, PlotArtifact>) {
  return Object.fromEntries(
    Object.entries(plots).map(([plotName, plot]) => [
      plotName,
      {
        traces: plot.data.map((trace) => summarizeTrace(plotName, trace)),
      },
    ]),
  );
}

/**
 * Resume a distribuição de valores numéricos em estatísticas agregadas e histograma em faixas.
 *
 * @param values - Vetor de valores numéricos a analisar.
 * @param maximumBins - Quantidade máxima de faixas (bins) a gerar. Padrão 10.
 * @returns Resumo contendo contagem, estatísticas descritivas e contagens por bin.
 */
export function summarizeNumericDistribution(
  values: number[],
  maximumBins = 10,
): NumericDistribution {
  const numeric = values
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  if (!numeric.length) return { count: 0, statistics: null, bins: [] };

  const minimum = numeric[0];
  const maximum = numeric[numeric.length - 1];
  const binCount = Math.min(
    maximumBins,
    Math.max(1, Math.ceil(Math.sqrt(numeric.length))),
  );
  if (minimum === maximum) {
    return {
      count: numeric.length,
      statistics: numericStatistics(numeric),
      bins: [{ lower: minimum, upper: maximum, count: numeric.length }],
    };
  }

  const width = (maximum - minimum) / binCount;
  const counts = Array.from({ length: binCount }, () => 0);
  for (const value of numeric) {
    const index = Math.min(Math.floor((value - minimum) / width), binCount - 1);
    counts[index] += 1;
  }
  return {
    count: numeric.length,
    statistics: numericStatistics(numeric),
    bins: counts.map((count, index) => ({
      lower: minimum + width * index,
      upper: minimum + width * (index + 1),
      count,
    })),
  };
}

/**
 * Sumariza o histórico de tentativas (trials) de ajuste de hiperparâmetros.
 *
 * @param trials - Lista de tentativas com status, valor da métrica e parâmetros.
 * @returns Estatísticas consolidadas de status, métricas e parâmetros explorados.
 */
export function summarizeTuningTrials(trials: TuningTrialContext[]) {
  const statusCounts: Record<string, number> = {};
  const parameterNames = new Set<string>();
  const metricValues: number[] = [];

  for (const trial of trials) {
    statusCounts[trial.status] = (statusCounts[trial.status] ?? 0) + 1;
    Object.keys(trial.parameters ?? {}).forEach((name) =>
      parameterNames.add(name),
    );
    if (
      typeof trial.metric_value === "number" &&
      Number.isFinite(trial.metric_value)
    ) {
      metricValues.push(trial.metric_value);
    }
  }

  return {
    total_count: trials.length,
    status_counts: statusCounts,
    metric_value_statistics: metricValues.length
      ? numericStatistics(metricValues)
      : null,
    parameter_names: [...parameterNames].sort(),
  };
}

/**
 * Converte um traço individual de plotagem em formato sumarizado segundo seu tipo.
 *
 * @param plotName - Nome identificador do gráfico (ex: shap_summary, confusion_matrix).
 * @param trace - Estrutura de dados do traço Plotly.
 * @returns Registro de dados agregados do traço.
 */
function summarizeTrace(plotName: string, trace: Plotly.Data) {
  const values = trace as unknown as Record<string, unknown>;
  const summary: Record<string, unknown> = {
    type: typeof values.type === "string" ? values.type : "scatter",
    name: typeof values.name === "string" ? values.name : null,
    point_count: Math.max(
      valueLength(values.x),
      valueLength(values.y),
      valueLength(values.z),
    ),
  };

  if (EXACT_AGGREGATE_PLOTS.has(plotName)) {
    summary.aggregation = "displayed_aggregate_values";
    for (const axis of ["x", "y", "z"] as const) {
      const arr = extractArray(values[axis]);
      if (arr.length > 0) summary[axis] = arr;
    }
    return summary;
  }
  if (HISTOGRAM_PLOTS.has(plotName)) {
    summary.aggregation = "histogram_and_distribution_statistics";
    summary.distribution = summarizeNumericDistribution(toNumbers(values.x));
    return summary;
  }
  if (PAIRED_SUMMARY_PLOTS.has(plotName)) {
    summary.aggregation = "paired_numeric_statistics";
    summary.paired = pairedNumericSummary(
      values.x,
      values.y,
      plotName === "actual_vs_predicted",
    );
    return summary;
  }
  if (plotName === "shap_summary") {
    const features = categoricalNumericSummary(values.y, values.x).sort(
      (left, right) =>
        right.mean_absolute_shap - left.mean_absolute_shap ||
        left.feature.localeCompare(right.feature),
    );
    summary.aggregation = "per_feature_shap_statistics";
    summary.feature_count = features.length;
    summary.features = features.slice(0, MAX_SHAP_FEATURES_PER_TRACE);
    summary.omitted_feature_count = Math.max(
      0,
      features.length - MAX_SHAP_FEATURES_PER_TRACE,
    );
    return summary;
  }
  summary.aggregation = "trace_metadata_only";
  return summary;
}

/**
 * Calcula resumo estatístico bivariado para gráficos pareados (ex: actual vs predicted).
 *
 * @param xValue - Valores do eixo X.
 * @param yValue - Valores do eixo Y.
 * @param includeDifference - Se deve calcular a média da diferença Y - X.
 * @returns Estrutura com contagem, estatísticas de X e Y, correlação e resíduo médio.
 */
function pairedNumericSummary(
  xValue: unknown,
  yValue: unknown,
  includeDifference: boolean,
) {
  const xArr = extractArray(xValue);
  const yArr = extractArray(yValue);
  if (!xArr.length || !yArr.length) return null;
  const pairs = xArr
    .map((x, index) => [Number(x), Number(yArr[index])] as const)
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!pairs.length) return null;
  const x = pairs.map((pair) => pair[0]);
  const y = pairs.map((pair) => pair[1]);
  return {
    count: pairs.length,
    x: numericStatistics(x),
    y: numericStatistics(y),
    correlation: pearsonCorrelation(x, y),
    ...(includeDifference
      ? { mean_y_minus_x: mean(y.map((value, index) => value - x[index])) }
      : {}),
  };
}

/**
 * Agrupa valores numéricos contínuos por categoria categórica (ex: valores SHAP por feature).
 *
 * @param categories - Array de categorias/features.
 * @param numericValues - Array de valores numéricos associados.
 * @returns Lista agregada de métricas por categoria.
 */
function categoricalNumericSummary(
  categories: unknown,
  numericValues: unknown,
) {
  const catArr = extractArray(categories);
  const numArr = extractArray(numericValues);
  if (!catArr.length || !numArr.length) return [];
  const grouped = new Map<string, number[]>();
  catArr.forEach((category, index) => {
    const numeric = Number(numArr[index]);
    if (!Number.isFinite(numeric)) return;
    const key = String(category);
    grouped.set(key, [...(grouped.get(key) ?? []), numeric]);
  });
  return [...grouped.entries()].map(([feature, values]) => ({
    feature,
    count: values.length,
    mean_shap: mean(values),
    mean_absolute_shap: mean(values.map(Math.abs)),
    minimum: Math.min(...values),
    maximum: Math.max(...values),
    positive_count: values.filter((value) => value > 0).length,
    negative_count: values.filter((value) => value < 0).length,
  }));
}

/**
 * Computa estatísticas descritivas básicas para um array de números.
 *
 * @param values - Array de valores numéricos.
 * @returns Mínimo, máximo, média, mediana, quartis e desvio padrão.
 */
function numericStatistics(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  const average = mean(sorted);
  return {
    minimum: sorted[0],
    maximum: sorted[sorted.length - 1],
    mean: average,
    median: quantile(sorted, 0.5),
    first_quartile: quantile(sorted, 0.25),
    third_quartile: quantile(sorted, 0.75),
    standard_deviation: Math.sqrt(
      mean(sorted.map((value) => (value - average) ** 2)),
    ),
  };
}

/**
 * Calcula o coeficiente de correlação linear de Pearson entre duas séries numéricas.
 *
 * @param x - Vetor de valores numéricos X.
 * @param y - Vetor de valores numéricos Y.
 * @returns Coeficiente de correlação no intervalo [-1, 1], ou null se variância nula.
 */
function pearsonCorrelation(x: number[], y: number[]): number | null {
  const xMean = mean(x);
  const yMean = mean(y);
  const numerator = x.reduce(
    (total, value, index) => total + (value - xMean) * (y[index] - yMean),
    0,
  );
  const denominator = Math.sqrt(
    x.reduce((total, value) => total + (value - xMean) ** 2, 0) *
      y.reduce((total, value) => total + (value - yMean) ** 2, 0),
  );
  return denominator === 0 ? null : numerator / denominator;
}

/**
 * Calcula o quantil linear para uma série numérica ordenada.
 *
 * @param sorted - Array previamente ordenado de números.
 * @param position - Posição do quantil desejado (entre 0 e 1).
 * @returns Valor interpolado do quantil.
 */
function quantile(sorted: number[], position: number): number {
  const index = (sorted.length - 1) * position;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

/**
 * Calcula a média aritmética de uma série numérica.
 *
 * @param values - Array de valores numéricos.
 * @returns Média aritmética dos valores.
 */
function mean(values: number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

/**
 * Extrai um array regular do JavaScript a partir de dados brutos ou codificados em base64.
 *
 * @param value - Objeto de dados, TypedArray, string base64 Plotly ou array simples.
 * @returns Array nativo com os elementos desempacotados.
 */
function extractArray(value: unknown): unknown[] {
  if (Array.isArray(value)) return value;
  if (ArrayBuffer.isView(value)) {
    return Array.from(value as unknown as ArrayLike<unknown>);
  }
  if (
    value &&
    typeof value === "object" &&
    "bdata" in value &&
    typeof (value as { bdata?: unknown }).bdata === "string"
  ) {
    const bdata = (value as { bdata: string }).bdata;
    const dtype = (value as { dtype?: unknown }).dtype;
    try {
      const binary =
        typeof atob === "function"
          ? atob(bdata)
          : typeof Buffer !== "undefined"
            ? Buffer.from(bdata, "base64").toString("binary")
            : "";
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      if (dtype === "f8" || dtype === "float64") {
        return Array.from(new Float64Array(bytes.buffer));
      }
      if (dtype === "f4" || dtype === "float32") {
        return Array.from(new Float32Array(bytes.buffer));
      }
      if (dtype === "i4" || dtype === "int32") {
        return Array.from(new Int32Array(bytes.buffer));
      }
      if (dtype === "i2" || dtype === "int16") {
        return Array.from(new Int16Array(bytes.buffer));
      }
      if (dtype === "i1" || dtype === "int8") {
        return Array.from(new Int8Array(bytes.buffer));
      }
      if (dtype === "u4" || dtype === "uint32") {
        return Array.from(new Uint32Array(bytes.buffer));
      }
      if (dtype === "u2" || dtype === "uint16") {
        return Array.from(new Uint16Array(bytes.buffer));
      }
      if (dtype === "u1" || dtype === "uint8") {
        return Array.from(new Uint8Array(bytes.buffer));
      }
      return Array.from(new Float64Array(bytes.buffer));
    } catch {
      return [];
    }
  }
  return [];
}

/**
 * Converte qualquer estrutura de dados em array de números finitos válidos.
 *
 * @param value - Valor ou coleção de valores arbitrários.
 * @returns Array de números filtrados.
 */
function toNumbers(value: unknown): number[] {
  const arr = extractArray(value);
  return arr.map(Number).filter(Number.isFinite);
}

/**
 * Obtém a contagem de elementos de uma estrutura de dados arbitrária.
 *
 * @param value - Valor a ser avaliado.
 * @returns Quantidade de itens presentes.
 */
function valueLength(value: unknown): number {
  return extractArray(value).length;
}
