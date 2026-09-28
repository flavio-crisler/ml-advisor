import { useQuery } from "@tanstack/react-query";
import { useCallback, useState, type ReactNode } from "react";
import Plot from "react-plotly.js";
import { useTranslation } from "react-i18next";
import { Link, useParams } from "react-router-dom";

import { AIChartExplanation } from "../../components/ai-chart-explanation";
import { AIExplanation } from "../../components/ai-explanation";
import { Card } from "../../components/ui/card";
import { HelpLabel } from "../../components/ui/info-tooltip";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkspaceLayout } from "../../components/workspace-header";
import {
  summarizeNumericDistribution,
  type ExplanationChartContext,
} from "../../lib/explanation-context";
import {
  getCalibration,
  getDecisionComparison,
  getDecisionRun,
  type ComparedApproach,
  type ComparedModel,
} from "../../services/decision-runs";
import { listExperiments } from "../../services/experiments";
import type { ResultExplanation } from "../../services/explanations";
import { DecisionLabNav } from "../decision-providers/decision-lab-nav";
import { HowItWorksButton } from "../../components/how-it-works";

type ReliabilityBin = {
  lower: number;
  upper: number;
  count: number;
  mean_confidence: number;
  empirical_accuracy: number;
};

type ThresholdPoint = {
  threshold: number;
  coverage: number;
  accuracy: number | null;
  error_rate: number | null;
  review_rate: number;
  estimated_cost: number | null;
};

type CalibrationMetrics = {
  brier_score: number;
  log_loss: number;
  expected_calibration_error: number;
  maximum_calibration_error: number;
  confidence_mean: number;
  reliability: ReliabilityBin[];
  confidence_histogram: Array<{ lower: number; upper: number; count: number }>;
  thresholds: ThresholdPoint[];
  latency_ms: number[];
};

/**
 * Página do Laboratório de Calibração para análise de confiabilidade, ECE e comparação com modelos supervisionados.
 *
 * @returns Página completa com diagramas de confiabilidade, curvas de custo/revisão e explicabilidade AI.
 */
export function CalibrationLabPage() {
  const { runId } = useParams();
  const { t } = useTranslation("decisionLab");
  const [threshold, setThreshold] = useState(0.8);
  const [experimentId, setExperimentId] = useState("");
  const [globalExplanation, setGlobalExplanation] = useState<
    ResultExplanation | undefined
  >();
  const handleExplanationChange = useCallback(
    (explanation: ResultExplanation | undefined) =>
      setGlobalExplanation(explanation),
    [],
  );
  const run = useQuery({
    queryKey: ["decision-runs", runId],
    queryFn: () => getDecisionRun(runId!),
    enabled: Boolean(runId),
    refetchInterval: (query) =>
      ["queued", "running"].includes(query.state.data?.status ?? "")
        ? 1000
        : false,
  });
  const calibration = useQuery({
    queryKey: ["decision-runs", runId, "calibration"],
    queryFn: () => getCalibration(runId!),
    enabled: run.data?.status === "completed",
  });
  const experiments = useQuery({
    queryKey: ["experiments"],
    queryFn: listExperiments,
  });
  const compatibleExperiments =
    experiments.data?.filter(
      (experiment) =>
        experiment.status === "completed" &&
        experiment.dataset_id === run.data?.dataset_id &&
        experiment.spec.target === run.data?.spec.target &&
        experiment.spec.task === run.data?.spec.task,
    ) ?? [];
  const comparison = useQuery({
    queryKey: ["decision-runs", runId, "comparison", experimentId],
    queryFn: () => getDecisionComparison(runId!, experimentId),
    enabled: Boolean(runId && experimentId),
    retry: false,
  });

  if (
    run.isPending ||
    (run.data?.status === "completed" && calibration.isPending)
  ) {
    return (
      <WorkspaceLayout>
        <DecisionLabNav />
        <StatePanel title={t("calibration.loading")} />
      </WorkspaceLayout>
    );
  }
  if (run.isError || !run.data) {
    return (
      <WorkspaceLayout>
        <DecisionLabNav />
        <StatePanel
          title={t("calibration.error")}
          description={t("calibration.errorDescription")}
          tone="error"
          actionLabel={t("states.retry")}
          onAction={() => void run.refetch()}
        />
      </WorkspaceLayout>
    );
  }
  if (run.data.status !== "completed") {
    const progress = run.data.row_count
      ? (run.data.processed_rows * 100) / run.data.row_count
      : 0;
    const failed = run.data.status === "failed";
    const cancelled = run.data.status === "cancelled";
    return (
      <WorkspaceLayout>
        <DecisionLabNav />
        <StatePanel
          title={
            failed
              ? t("calibration.failed")
              : cancelled
                ? t("calibration.cancelled")
                : t("calibration.processing")
          }
          description={`${progress.toFixed(1)}% · ${run.data.processed_rows}/${run.data.row_count}${failed && run.data.error_message ? ` · ${run.data.error_message === "decision_job_interrupted" ? t("calibration.interrupted") : run.data.error_message}` : ""}`}
          tone={failed ? "error" : "neutral"}
          actionLabel={failed || cancelled ? t("calibration.back") : undefined}
          onAction={
            failed || cancelled ? () => window.history.back() : undefined
          }
        />
      </WorkspaceLayout>
    );
  }
  if (calibration.isError || !calibration.data) {
    return (
      <WorkspaceLayout>
        <DecisionLabNav />
        <StatePanel
          title={t("calibration.error")}
          description={t("calibration.errorDescription")}
          tone="error"
          actionLabel={t("states.retry")}
          onAction={() => void calibration.refetch()}
        />
      </WorkspaceLayout>
    );
  }

  const metrics = calibration.data.metrics as unknown as CalibrationMetrics;
  // Older artifacts encoded cache hits as zero-millisecond provider calls.
  // Omitting those synthetic zeros keeps the provider-latency chart honest.
  const providerLatencies = metrics.latency_ms.filter((latency) => latency > 0);
  const latencyDistribution = summarizeNumericDistribution(providerLatencies);
  const current =
    metrics.thresholds.reduce((closest, point) =>
      Math.abs(point.threshold - threshold) <
      Math.abs(closest.threshold - threshold)
        ? point
        : closest,
    ) ?? metrics.thresholds[0];
  const plotLayout = {
    autosize: true,
    margin: { l: 50, r: 20, t: 20, b: 45 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
  };
  const axes = (
    x: string,
    y: string,
    unitInterval = false,
  ): Partial<Plotly.Layout> => ({
    ...plotLayout,
    xaxis: {
      title: { text: x },
      ...(unitInterval ? { range: [0, 1] } : {}),
    },
    yaxis: {
      title: { text: y },
      ...(unitInterval ? { range: [0, 1] } : {}),
    },
  });
  const explanationBaseContext = {
    decision_run: {
      provider_id: run.data.provider_id,
      requested_model: run.data.requested_model,
      resolved_model: run.data.resolved_model,
      task: run.data.spec.task,
      question: run.data.spec.question.instructions,
      row_count: run.data.row_count,
      processed_rows: run.data.processed_rows,
      request_count: run.data.request_count,
      cache_hits: run.data.cache_hits,
      input_tokens: run.data.input_tokens,
      estimated_cost: run.data.estimated_cost,
      latency_ms_total: run.data.latency_ms_total,
      metrics: run.data.metrics ?? {},
    },
    calibration_summary: {
      brier_score: metrics.brier_score,
      log_loss: metrics.log_loss,
      expected_calibration_error: metrics.expected_calibration_error,
      maximum_calibration_error: metrics.maximum_calibration_error,
      confidence_mean: metrics.confidence_mean,
    },
    selected_threshold: threshold,
    selected_threshold_metrics: current,
    supervised_comparison: comparison.data
      ? {
          experiment_name: comparison.data.experiment_name,
          contract: comparison.data.contract,
          unavailable_models: comparison.data.unavailable_models.map(
            ({ model_id, reason }) => ({ model_id, reason }),
          ),
          models: comparison.data.models.map(
            ({
              model_id,
              run_index,
              sample_size,
              classes,
              supervised,
              zero_shot,
              metric_deltas,
              disagreement,
            }) => ({
              model_id,
              run_index,
              sample_size,
              classes,
              supervised,
              zero_shot,
              metric_deltas,
              disagreement,
            }),
          ),
        }
      : null,
  };
  const explanationCharts: ExplanationChartContext[] = [
    {
      chart_id: "reliability",
      title: t("calibration.reliability"),
      purpose: t("calibration.help.reliability"),
      axes: {
        x: t("calibration.confidenceAxis"),
        y: t("calibration.accuracyAxis"),
      },
      data_policy: "aggregate_only_no_raw_rows",
      data: {
        bins: metrics.reliability,
        ideal_reference: { x: [0, 1], y: [0, 1] },
      },
    },
    {
      chart_id: "confidence_histogram",
      title: t("calibration.histogram"),
      purpose: t("calibration.help.histogram"),
      axes: {
        x: t("calibration.confidenceAxis"),
        y: t("calibration.sampleCountAxis"),
      },
      data_policy: "aggregate_only_no_raw_rows",
      data: { bins: metrics.confidence_histogram },
    },
    {
      chart_id: "threshold_tradeoff",
      title: t("calibration.thresholdChart"),
      purpose: t("calibration.help.thresholdChart"),
      axes: {
        x: t("calibration.thresholdAxis"),
        y: t("calibration.rateAxis"),
      },
      data_policy: "aggregate_only_no_raw_rows",
      data: {
        points: metrics.thresholds.map(({ threshold, coverage, accuracy }) => ({
          threshold,
          coverage,
          accuracy,
        })),
      },
    },
    {
      chart_id: "provider_latency",
      title: t("calibration.latency"),
      purpose: t("calibration.help.latency"),
      axes: {
        x: t("calibration.latencyAxis"),
        y: t("calibration.sampleCountAxis"),
      },
      data_policy: "aggregate_only_no_raw_rows",
      data: latencyDistribution,
    },
  ];
  const explanationContext = {
    ...explanationBaseContext,
    charts: explanationCharts,
  };

  return (
    <WorkspaceLayout>
      <DecisionLabNav />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link className="text-sm text-primary" to="/decision-lab/benchmark">
            {t("calibration.back")}
          </Link>
          <h1 className="mt-3 text-4xl font-semibold">
            {t("calibration.title")}
          </h1>
          <p className="mt-2 text-muted">
            {run.data.spec.question.instructions}
          </p>
        </div>
        <HowItWorksButton screen="calibrationLab" />
      </section>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Metric
          label={t("calibration.brier")}
          help={t("calibration.help.brier")}
          value={metrics.brier_score}
        />
        <Metric
          label={t("calibration.logLoss")}
          help={t("calibration.help.logLoss")}
          value={metrics.log_loss}
        />
        <Metric
          label={t("calibration.ece")}
          help={t("calibration.help.ece")}
          value={metrics.expected_calibration_error}
        />
        <Metric
          label={t("calibration.latencyP95")}
          help={t("calibration.help.latencyP95")}
          value={run.data.metrics?.latency_p95_ms}
        />
        <Metric
          label={t("calibration.providerCalls")}
          help={t("calibration.help.providerCalls")}
          value={run.data.request_count}
          digits={0}
        />
        <Metric
          label={t("calibration.cacheHits")}
          help={t("calibration.help.cacheHits")}
          value={run.data.cache_hits}
          digits={0}
        />
        <Metric
          label={t("calibration.runCost")}
          help={t("calibration.help.runCost")}
          value={run.data.estimated_cost}
          digits={6}
        />
      </div>

      <Card>
        <label className="font-semibold" htmlFor="confidence-threshold">
          <HelpLabel help={t("calibration.help.threshold")}>
            {t("calibration.threshold")}
          </HelpLabel>
          : {threshold.toFixed(2)}
        </label>
        <input
          id="confidence-threshold"
          className="mt-4 w-full accent-primary"
          type="range"
          min="0.5"
          max="1"
          step="0.05"
          value={threshold}
          onChange={(event) => setThreshold(Number(event.target.value))}
        />
        <div className="mt-5 grid gap-4 sm:grid-cols-3">
          <MetricValue
            label={t("calibration.coverage")}
            help={t("calibration.help.coverage")}
            value={current.coverage}
            percent
          />
          <MetricValue
            label={t("calibration.accuracy")}
            help={t("calibration.help.accuracy")}
            value={current.accuracy}
            percent
          />
          <MetricValue
            label={t("calibration.reviewRate")}
            help={t("calibration.help.reviewRate")}
            value={current.review_rate}
            percent
          />
        </div>
        <p className="mt-4 text-xs text-muted">
          {t("calibration.persistedOnly")}
        </p>
      </Card>

      <ComparisonSection
        experiments={compatibleExperiments}
        experimentId={experimentId}
        onExperimentChange={setExperimentId}
        comparison={comparison.data}
        loading={comparison.isPending && Boolean(experimentId)}
        error={comparison.isError}
      />

      <AIExplanation
        scope="decision_calibration"
        context={explanationContext}
        onExplanationChange={handleExplanationChange}
        improvementSource={
          experimentId && comparison.data ? { experimentId } : undefined
        }
      />

      <div className="grid gap-5 xl:grid-cols-2">
        <Chart
          title={t("calibration.reliability")}
          help={t("calibration.help.reliability")}
        >
          <Plot
            data={[
              {
                x: metrics.reliability.map((bin) => bin.mean_confidence),
                y: metrics.reliability.map((bin) => bin.empirical_accuracy),
                type: "scatter",
                mode: "lines+markers",
                name: t("calibration.observed"),
              },
              {
                x: [0, 1],
                y: [0, 1],
                type: "scatter",
                mode: "lines",
                name: t("calibration.ideal"),
              },
            ]}
            layout={axes(
              t("calibration.confidenceAxis"),
              t("calibration.accuracyAxis"),
              true,
            )}
            useResizeHandler
            className="h-72 w-full"
            config={{ displayModeBar: false, responsive: true }}
          />
          <AIChartExplanation
            chartId="reliability"
            context={{
              ...explanationBaseContext,
              charts: [
                explanationCharts.find(
                  (chart) => chart.chart_id === "reliability",
                )!,
              ],
            }}
            globalExplanation={globalExplanation?.chart_explanations?.find(
              (item) => item.chart_id === "reliability",
            )}
          />
        </Chart>
        <Chart
          title={t("calibration.histogram")}
          help={t("calibration.help.histogram")}
        >
          <Plot
            data={[
              {
                x: metrics.confidence_histogram.map(
                  (bin) => (bin.lower + bin.upper) / 2,
                ),
                y: metrics.confidence_histogram.map((bin) => bin.count),
                type: "bar",
              },
            ]}
            layout={axes(
              t("calibration.confidenceAxis"),
              t("calibration.sampleCountAxis"),
            )}
            useResizeHandler
            className="h-72 w-full"
            config={{ displayModeBar: false, responsive: true }}
          />
          <AIChartExplanation
            chartId="confidence_histogram"
            context={{
              ...explanationBaseContext,
              charts: [
                explanationCharts.find(
                  (chart) => chart.chart_id === "confidence_histogram",
                )!,
              ],
            }}
            globalExplanation={globalExplanation?.chart_explanations?.find(
              (item) => item.chart_id === "confidence_histogram",
            )}
          />
        </Chart>
        <Chart
          title={t("calibration.thresholdChart")}
          help={t("calibration.help.thresholdChart")}
        >
          <Plot
            data={[
              {
                x: metrics.thresholds.map((point) => point.threshold),
                y: metrics.thresholds.map((point) => point.coverage),
                type: "scatter",
                mode: "lines+markers",
                name: t("calibration.coverage"),
              },
              {
                x: metrics.thresholds.map((point) => point.threshold),
                y: metrics.thresholds.map((point) => point.accuracy),
                type: "scatter",
                mode: "lines+markers",
                name: t("calibration.accuracy"),
              },
            ]}
            layout={axes(
              t("calibration.thresholdAxis"),
              t("calibration.rateAxis"),
              true,
            )}
            useResizeHandler
            className="h-72 w-full"
            config={{ displayModeBar: false, responsive: true }}
          />
          <AIChartExplanation
            chartId="threshold_tradeoff"
            context={{
              ...explanationBaseContext,
              charts: [
                explanationCharts.find(
                  (chart) => chart.chart_id === "threshold_tradeoff",
                )!,
              ],
            }}
            globalExplanation={globalExplanation?.chart_explanations?.find(
              (item) => item.chart_id === "threshold_tradeoff",
            )}
          />
        </Chart>
        {providerLatencies.length ? (
          <Chart
            title={t("calibration.latency")}
            help={t("calibration.help.latency")}
          >
            <Plot
              data={[
                {
                  x: latencyDistribution.bins.map(
                    (bin) => (bin.lower + bin.upper) / 2,
                  ),
                  y: latencyDistribution.bins.map((bin) => bin.count),
                  type: "bar",
                },
              ]}
              layout={axes(
                t("calibration.latencyAxis"),
                t("calibration.sampleCountAxis"),
              )}
              useResizeHandler
              className="h-72 w-full"
              config={{ displayModeBar: false, responsive: true }}
            />
            <AIChartExplanation
              chartId="provider_latency"
              context={{
                ...explanationBaseContext,
                charts: [
                  explanationCharts.find(
                    (chart) => chart.chart_id === "provider_latency",
                  )!,
                ],
              }}
              globalExplanation={globalExplanation?.chart_explanations?.find(
                (item) => item.chart_id === "provider_latency",
              )}
            />
          </Chart>
        ) : (
          <Chart
            title={t("calibration.latency")}
            help={t("calibration.help.latency")}
          >
            <p className="mt-4 text-sm text-muted">
              {t("calibration.noProviderLatency")}
            </p>
            <AIChartExplanation
              chartId="provider_latency"
              context={{
                ...explanationBaseContext,
                charts: [
                  explanationCharts.find(
                    (chart) => chart.chart_id === "provider_latency",
                  )!,
                ],
              }}
              globalExplanation={globalExplanation?.chart_explanations?.find(
                (item) => item.chart_id === "provider_latency",
              )}
            />
          </Chart>
        )}
      </div>
    </WorkspaceLayout>
  );
}

function ComparisonSection({
  experiments,
  experimentId,
  onExperimentChange,
  comparison,
  loading,
  error,
}: {
  experiments: Awaited<ReturnType<typeof listExperiments>>;
  experimentId: string;
  onExperimentChange: (value: string) => void;
  comparison: Awaited<ReturnType<typeof getDecisionComparison>> | undefined;
  loading: boolean;
  error: boolean;
}) {
  const { t } = useTranslation("decisionLab");
  return (
    <Card>
      <h2 className="text-xl font-semibold">
        <HelpLabel help={t("comparison.help.title")}>
          {t("comparison.title")}
        </HelpLabel>
      </h2>
      <p className="mt-2 text-sm text-muted">{t("comparison.pairedTooltip")}</p>
      <label className="mt-5 block max-w-xl text-sm font-medium">
        <span className="mb-2 block">{t("comparison.experiment")}</span>
        <select
          className="w-full rounded-lg border border-border bg-background px-3 py-2"
          value={experimentId}
          onChange={(event) => onExperimentChange(event.target.value)}
        >
          <option value="">{t("comparison.selectExperiment")}</option>
          {experiments.map((experiment) => (
            <option key={experiment.id} value={experiment.id}>
              {experiment.name}
            </option>
          ))}
        </select>
      </label>
      {!experiments.length ? (
        <p className="mt-4 text-sm text-muted">
          {t("comparison.noCompatibleExperiments")}
        </p>
      ) : null}
      {loading ? (
        <p className="mt-4 text-sm text-muted">{t("comparison.loading")}</p>
      ) : null}
      {error ? (
        <p className="mt-4 text-sm text-error">{t("comparison.error")}</p>
      ) : null}
      {comparison?.models.length ? (
        <div className="mt-5 space-y-6">
          {comparison.models.map((model) => (
            <ComparedModelTable key={model.model_run_id} model={model} />
          ))}
        </div>
      ) : null}
      {comparison && !comparison.models.length ? (
        <p className="mt-4 text-sm text-muted">
          {t("comparison.noPersistedPredictions")}
        </p>
      ) : null}
      {comparison?.unavailable_models.length ? (
        <p className="mt-4 text-xs text-warning">
          {t("comparison.unavailableModels", {
            models: comparison.unavailable_models
              .map((item) => item.model_id)
              .join(", "),
          })}
        </p>
      ) : null}
    </Card>
  );
}

function ComparedModelTable({ model }: { model: ComparedModel }) {
  const { t } = useTranslation("decisionLab");
  const rows: Array<{
    key: string;
    label: ReactNode;
    metrics: ComparedApproach["metrics"];
    latency: number | null | undefined;
    cost: number | null | undefined;
  }> = [
    {
      key: "supervised",
      label: `${t("comparison.supervised")} · ${model.model_id}`,
      metrics: model.supervised.metrics,
      latency: model.supervised.latency_p95_ms,
      cost: model.supervised.estimated_cost,
    },
    {
      key: "zero-shot",
      label: t("comparison.zeroShot"),
      metrics: model.zero_shot.metrics,
      latency: model.zero_shot.latency_p95_ms,
      cost: model.zero_shot.estimated_cost,
    },
    {
      key: "delta",
      label: (
        <HelpLabel help={t("comparison.help.delta")}>
          {t("comparison.delta")}
        </HelpLabel>
      ),
      metrics: model.metric_deltas,
      latency: null,
      cost: null,
    },
  ];
  return (
    <section>
      <div className="flex flex-wrap items-end justify-between gap-2">
        <h3 className="font-semibold">
          {model.model_id} · {t("comparison.run", { index: model.run_index })}
        </h3>
        <p className="text-xs text-muted">
          {t("comparison.sameRows", { count: model.sample_size })}
        </p>
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[860px] text-left text-sm">
          <thead>
            <tr className="border-b border-border text-muted">
              <th className="pb-2">{t("comparison.approach")}</th>
              {[
                "accuracy",
                "f1",
                "rocAuc",
                "prAuc",
                "brier",
                "ece",
                "latency",
                "cost",
              ].map((key) => (
                <th key={key} className="pb-2">
                  <HelpLabel help={t(`comparison.help.${key}`)}>
                    {t(`comparison.columns.${key}`)}
                  </HelpLabel>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.key}
                className="border-b border-border last:border-0"
              >
                <td className="py-2 font-medium">{row.label}</td>
                <MetricCell value={row.metrics.accuracy} />
                <MetricCell value={row.metrics.f1} />
                <MetricCell value={row.metrics.roc_auc} />
                <MetricCell value={row.metrics.pr_auc} />
                <MetricCell value={row.metrics.brier_score} />
                <MetricCell value={row.metrics.expected_calibration_error} />
                <MetricCell value={row.latency} digits={2} />
                <MetricCell value={row.cost} digits={6} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-5">
        <ComparisonCount
          label={t("comparison.bothCorrect")}
          value={model.disagreement.both_correct}
        />
        <ComparisonCount
          label={t("comparison.supervisedOnly")}
          value={model.disagreement.supervised_only_correct}
        />
        <ComparisonCount
          label={t("comparison.zeroShotOnly")}
          value={model.disagreement.zero_shot_only_correct}
        />
        <ComparisonCount
          label={t("comparison.bothWrong")}
          value={model.disagreement.both_wrong}
        />
        <ComparisonCount
          label={t("comparison.disagreements")}
          value={model.disagreement.prediction_disagreements}
        />
      </dl>
    </section>
  );
}

/**
 * Célula de tabela formatada para exibição de valor de métrica numérica.
 *
 * @param props.value - Valor numérico ou indefinido da métrica.
 * @param props.digits - Quantidade de casas decimais a exibir. Padrão 4.
 * @returns Elemento td estilizado.
 */
function MetricCell({
  value,
  digits = 4,
}: {
  value: unknown;
  digits?: number;
}) {
  return (
    <td className="py-2 font-mono">
      {typeof value === "number" ? value.toFixed(digits) : "—"}
    </td>
  );
}

/**
 * Exibe contagem de concordância ou discrepância entre abordagens comparadas.
 *
 * @param props.label - Rótulo da contagem.
 * @param props.value - Quantidade de exemplos observados.
 * @returns Bloco de termo e definição dl.
 */
function ComparisonCount({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="text-muted">{label}</dt>
      <dd className="mt-1 font-mono font-semibold">{value}</dd>
    </div>
  );
}

/**
 * Card individual com métrica principal destacada e tooltip de ajuda explicativo.
 *
 * @param props.label - Nome da métrica.
 * @param props.help - Conteúdo de ajuda do tooltip.
 * @param props.value - Valor numérico da métrica.
 * @param props.digits - Número de casas decimais. Padrão 4.
 * @returns Card contendo a métrica formatada.
 */
function Metric({
  label,
  help,
  value,
  digits = 4,
}: {
  label: string;
  help: ReactNode;
  value: unknown;
  digits?: number;
}) {
  return (
    <Card>
      <p className="text-sm text-muted">
        <HelpLabel help={help}>{label}</HelpLabel>
      </p>
      <p className="mt-2 font-mono text-2xl font-semibold">
        {typeof value === "number" ? value.toFixed(digits) : "—"}
      </p>
    </Card>
  );
}

/**
 * Exibe valor numérico de métrica formatado com precisão decimal ou percentual.
 *
 * @param props.label - Nome da métrica.
 * @param props.help - Conteúdo de ajuda contextual.
 * @param props.value - Valor numérico ou nulo.
 * @param props.percent - Se deve exibir como porcentagem (ex: 85.2%). Padrão false.
 * @returns Bloco com rótulo e valor destacado.
 */
function MetricValue({
  label,
  help,
  value,
  percent = false,
}: {
  label: string;
  help: ReactNode;
  value: number | null;
  percent?: boolean;
}) {
  return (
    <div>
      <p className="text-xs text-muted">
        <HelpLabel help={help}>{label}</HelpLabel>
      </p>
      <p className="mt-1 font-mono text-xl font-semibold">
        {value == null
          ? "—"
          : percent
            ? `${(value * 100).toFixed(1)}%`
            : value.toFixed(4)}
      </p>
    </div>
  );
}

/**
 * Container padronizado de gráfico analítico com título e tooltip de ajuda.
 *
 * @param props.title - Título do gráfico.
 * @param props.help - Explicação conceitual exibida no tooltip de ajuda.
 * @param props.children - Elemento do gráfico ou componente de plotagem.
 * @returns Card contendo o cabeçalho e corpo do gráfico.
 */
function Chart({
  title,
  help,
  children,
}: {
  title: string;
  help: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card className="min-w-0 overflow-hidden">
      <h2 className="font-semibold">
        <HelpLabel help={help}>{title}</HelpLabel>
      </h2>
      {children}
    </Card>
  );
}
