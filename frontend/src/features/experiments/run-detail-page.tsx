import { useQuery } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import Plot from "react-plotly.js";
import { useTranslation } from "react-i18next";
import { useNavigate, useParams } from "react-router-dom";

import { AIChartExplanation } from "../../components/ai-chart-explanation";
import { AIExplanation } from "../../components/ai-explanation";
import { WorkspaceLayout } from "../../components/workspace-header";
import { StatusBadge } from "../../components/status-badge";
import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { HelpLabel } from "../../components/ui/info-tooltip";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkflowSteps } from "../../components/workflow-steps";
import { HowItWorksButton } from "../../components/how-it-works";
import {
  buildPlotExplanationCharts,
  summarizeTuningTrials,
} from "../../lib/explanation-context";
import type { ResultExplanation } from "../../services/explanations";
import {
  getModelRun,
  getRunAnalysis,
  getTuningTrials,
} from "../../services/experiments";

/**
 * Página de detalhes aprofundados de uma execução individual de modelo (model run).
 *
 * @returns Interface contendo métricas calculadas, gráficos interativos Plotly (SHAP, ROC, matriz de confusão) e explicações via IA.
 */
export function RunDetailPage() {
  const { experimentId, runId } = useParams();
  const { t } = useTranslation("experiments");
  const navigate = useNavigate();
  const [globalExplanation, setGlobalExplanation] = useState<
    ResultExplanation | undefined
  >();
  const handleExplanationChange = useCallback(
    (explanation: ResultExplanation | undefined) =>
      setGlobalExplanation(explanation),
    [],
  );
  const run = useQuery({
    queryKey: ["runs", runId],
    queryFn: () => getModelRun(runId!),
    enabled: Boolean(runId),
  });
  const trials = useQuery({
    queryKey: ["runs", runId, "trials"],
    queryFn: () => getTuningTrials(runId!),
    enabled: Boolean(runId),
  });
  const analysis = useQuery({
    queryKey: ["runs", runId, "analysis"],
    queryFn: () => getRunAnalysis(runId!),
    enabled: run.data?.status === "completed",
    retry: false,
  });

  if (run.isPending)
    return (
      <WorkspaceLayout>
        <StatePanel title={t("loading")} />
      </WorkspaceLayout>
    );
  if (run.isError || !run.data)
    return (
      <WorkspaceLayout>
        <StatePanel
          title={t("errors.load")}
          description={t("errors.loadDescription")}
          tone="error"
          actionLabel={t("actions.retry")}
          onAction={() => void run.refetch()}
        />
      </WorkspaceLayout>
    );

  const metrics = run.data.metrics?.values;
  const effective =
    run.data.metrics?.effective_parameters ?? run.data.parameters;
  const explanationBaseContext = {
    model_run: {
      model_id: run.data.model_id,
      run_index: run.data.run_index,
      status: run.data.status,
      duration_seconds: run.data.duration_seconds,
      parameters: effective,
      metrics: metrics ?? {},
    },
    tuning: summarizeTuningTrials(trials.data ?? []),
    analysis: analysis.data
      ? {
          task: analysis.data.task,
          explainability: {
            status: analysis.data.explainability.status,
            reason: analysis.data.explainability.reason,
            detail: analysis.data.explainability.detail,
          },
        }
      : null,
  };
  const explanationCharts = analysis.data
    ? buildPlotExplanationCharts(analysis.data.plots, (name) => {
        const axis = plotAxisMetadata(name);
        return {
          title: t(`plots.${name}`, { defaultValue: name }),
          purpose: t(`plotHelp.${name}`, { defaultValue: name }),
          axes: {
            x: axis ? t(axis.x) : "x",
            y: axis ? t(axis.y) : "y",
          },
        };
      })
    : [];
  const explanationContext = {
    ...explanationBaseContext,
    charts: explanationCharts,
  };
  return (
    <WorkspaceLayout>
      <WorkflowSteps current={6} />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <button
            className="text-sm text-primary"
            onClick={() => navigate(`/experiments/${experimentId}`)}
          >
            {t("actions.backToExperiment")}
          </button>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <h1 className="text-3xl font-semibold">{run.data.model_id}</h1>
            <StatusBadge status={run.data.status} />
          </div>
          <p className="mt-2 text-sm text-muted">
            {t("detail.runIndex", { index: run.data.run_index })}
          </p>
        </div>
        <HowItWorksButton screen="runDetail" />
      </section>

      {run.data.status === "failed" ? (
        <Card className="border-error/30 bg-error/5" role="alert">
          <h2 className="font-semibold text-error">{t("run.failureTitle")}</h2>
          <p className="mt-2 break-words font-mono text-sm">
            {run.data.error_message ?? t("run.failureUnknown")}
          </p>
          <p className="mt-3 text-sm text-muted">{t("run.failureHelp")}</p>
          <Button
            className="mt-4"
            variant="outline"
            onClick={() => navigate(`/experiments/${experimentId}`)}
          >
            {t("run.reviewExperiment")}
          </Button>
        </Card>
      ) : null}

      <nav className="flex flex-wrap gap-2" aria-label={t("run.sectionsLabel")}>
        {[
          ["overview", t("run.sections.overview")],
          ["models", t("run.sections.models")],
          ["metrics", t("run.sections.metrics")],
          ["charts", t("run.sections.charts")],
          ["parameters", t("run.sections.parameters")],
          ["artifacts", t("run.sections.artifacts")],
        ].map(([id, label]) => (
          <a
            key={id}
            href={`#${id}`}
            className="rounded-full border border-border bg-surface px-3 py-1.5 text-sm font-medium text-muted transition-colors hover:border-primary hover:text-primary"
          >
            {label}
          </a>
        ))}
      </nav>

      <section id="overview" className="grid gap-5 lg:grid-cols-2">
        <Card>
          <h2 className="text-lg font-semibold">
            {t("run.sections.overview")}
          </h2>
          <dl className="mt-4 grid gap-4 text-sm sm:grid-cols-2">
            <ResultField
              label={t("fields.status")}
              value={t(`status.${run.data.status}`, {
                defaultValue: run.data.status,
              })}
            />
            <ResultField
              label={t("detail.runIndex", { index: run.data.run_index })}
              value={run.data.id}
            />
            <ResultField
              label={t("run.startedAt")}
              value={run.data.started_at ?? t("detail.notAvailable")}
            />
            <ResultField
              label={t("run.duration")}
              value={
                run.data.duration_seconds == null
                  ? t("detail.notAvailable")
                  : t("run.durationValue", {
                      value: run.data.duration_seconds.toFixed(2),
                    })
              }
            />
          </dl>
        </Card>
        <Card id="models">
          <h2 className="text-lg font-semibold">{t("run.sections.models")}</h2>
          <p className="mt-4 font-mono text-lg font-semibold">
            {run.data.model_id}
          </p>
          <p className="mt-2 text-sm text-muted">{t("run.modelHelp")}</p>
        </Card>
      </section>

      <JsonCard id="metrics" title={t("run.metrics")} value={metrics ?? {}} />

      {trials.data?.length ? (
        <Card className="overflow-x-auto">
          <h2 className="mb-4 text-lg font-semibold">
            {t("run.tuningTrials")}
          </h2>
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead>
              <tr className="border-b border-border text-muted">
                <th className="pb-3">#</th>
                <th className="pb-3">{t("fields.status")}</th>
                <th className="pb-3">{t("run.metricValue")}</th>
                <th className="pb-3">{t("comparison.parameters")}</th>
              </tr>
            </thead>
            <tbody>
              {trials.data.map((trial) => (
                <tr
                  key={trial.id}
                  className="border-b border-border last:border-0"
                >
                  <td className="py-3 font-mono">{trial.trial_number + 1}</td>
                  <td className="py-3">
                    {t(`status.${trial.status}`, {
                      defaultValue: trial.status,
                    })}
                  </td>
                  <td className="py-3 font-mono">
                    {trial.metric_value?.toFixed(5) ?? "-"}
                  </td>
                  <td className="py-3 font-mono text-xs">
                    {JSON.stringify(trial.parameters)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ) : null}

      {analysis.data ? (
        <section id="charts">
          <h2 className="mb-4 text-xl font-semibold">{t("run.analysis")}</h2>
          <div className="grid gap-5 xl:grid-cols-2">
            {Object.entries(analysis.data.plots).map(([name, plot]) => {
              const chartContext = explanationCharts.find(
                (chart) => chart.chart_id === name,
              )!;
              return (
                <Card key={name} className="min-w-0 overflow-hidden">
                  <h3 className="mb-3 font-semibold">
                    <HelpLabel help={t(`plotHelp.${name}`)}>
                      {t(`plots.${name}`, { defaultValue: name })}
                    </HelpLabel>
                  </h3>
                  <Plot
                    data={localizeTraces(plot.data, t)}
                    layout={{
                      ...plot.layout,
                      ...localizedAxes(name, plot.layout, t),
                      autosize: true,
                      margin: { l: 55, r: 20, t: 20, b: 50 },
                    }}
                    config={{ displaylogo: false, responsive: true }}
                    useResizeHandler
                    className="h-[360px] w-full"
                  />
                  <AIChartExplanation
                    chartId={name}
                    context={{
                      ...explanationBaseContext,
                      charts: [chartContext],
                    }}
                    globalExplanation={globalExplanation?.chart_explanations?.find(
                      (item) => item.chart_id === name,
                    )}
                  />
                </Card>
              );
            })}
          </div>
          {analysis.data.explainability.status === "unavailable" ? (
            <Card className="mt-5">
              <p className="text-warning">{t("run.shapUnavailable")}</p>
            </Card>
          ) : null}
        </section>
      ) : analysis.isError ? (
        <Card id="charts">
          <p className="text-muted">{t("run.analysisUnavailable")}</p>
        </Card>
      ) : null}

      <AIExplanation
        scope="model_run"
        context={explanationContext}
        onExplanationChange={handleExplanationChange}
        improvementSource={
          experimentId ? { experimentId, modelRunId: run.data.id } : undefined
        }
      />

      <JsonCard id="parameters" title={t("run.parameters")} value={effective} />

      <Card id="artifacts">
        <h2 className="text-lg font-semibold">{t("run.sections.artifacts")}</h2>
        <p className="mt-2 text-sm text-muted">{t("run.artifactsHelp")}</p>
        <code className="mt-4 block overflow-x-auto rounded-lg bg-code p-4 text-xs text-code-text">
          {run.data.artifact_path ?? t("detail.notAvailable")}
        </code>
      </Card>
    </WorkspaceLayout>
  );
}

/**
 * Traduz rótulos e metadados de traços de plotagem para o idioma da interface.
 *
 * @param data - Array de traços Plotly.
 * @param t - Função de tradução i18n.
 * @returns Array de traços com legendas traduzidas.
 */
function localizeTraces(
  data: Plotly.Data[],
  t: (key: string, options?: Record<string, unknown>) => string,
): Plotly.Data[] {
  return data.map((trace) => {
    const meta = (trace as { meta?: unknown }).meta as
      { role?: string; label?: string } | null | undefined;
    if (meta?.role !== "baseline") return trace;
    return {
      ...trace,
      name: meta.label
        ? `${meta.label} · ${t("plotAxes.baseline")}`
        : t("plotAxes.baseline"),
    } as Plotly.Data;
  });
}

/**
 * Sobrescreve os títulos dos eixos X e Y do layout Plotly com rótulos localizados.
 *
 * @param name - Nome identificador do gráfico.
 * @param layout - Layout parcial Plotly original.
 * @param t - Função de tradução i18n.
 * @returns Layout com títulos de eixos ajustados.
 */
function localizedAxes(
  name: string,
  layout: Partial<Plotly.Layout>,
  t: (key: string) => string,
): Partial<Plotly.Layout> {
  const selected = plotAxisMetadata(name);
  if (!selected) return {};
  return {
    xaxis: {
      ...(layout.xaxis as Partial<Plotly.LayoutAxis>),
      title: { text: t(selected.x) },
      ...(selected.unitInterval ? { range: [0, 1] } : {}),
    },
    yaxis: {
      ...(layout.yaxis as Partial<Plotly.LayoutAxis>),
      title: { text: t(selected.y) },
      ...(selected.unitInterval ? { range: [0, 1] } : {}),
    },
  };
}

/**
 * Retorna as chaves de tradução dos eixos para um gráfico específico.
 *
 * @param name - Identificador do tipo de gráfico.
 * @returns Objeto com chaves i18n para x, y e flag de intervalo unitário, ou undefined.
 */
function plotAxisMetadata(
  name: string,
): { x: string; y: string; unitInterval?: boolean } | undefined {
  const axes: Record<string, { x: string; y: string; unitInterval?: boolean }> =
    {
      confusion_matrix: {
        x: "plotAxes.predictedClass",
        y: "plotAxes.actualClass",
      },
      roc_curve: {
        x: "plotAxes.falsePositiveRate",
        y: "plotAxes.truePositiveRate",
        unitInterval: true,
      },
      precision_recall_curve: {
        x: "plotAxes.recall",
        y: "plotAxes.precision",
        unitInterval: true,
      },
      probability_distribution: {
        x: "plotAxes.probability",
        y: "plotAxes.count",
      },
      actual_vs_predicted: {
        x: "plotAxes.actualValue",
        y: "plotAxes.predictedValue",
      },
      residual_plot: {
        x: "plotAxes.predictedValue",
        y: "plotAxes.residual",
      },
      residual_distribution: {
        x: "plotAxes.residual",
        y: "plotAxes.count",
      },
      feature_importance: {
        x: "plotAxes.meanAbsoluteShap",
        y: "plotAxes.feature",
      },
      shap_summary: {
        x: "plotAxes.shapValue",
        y: "plotAxes.feature",
      },
      learning_curve: {
        x: "plotAxes.trainingRows",
        y: "plotAxes.validationMetric",
      },
    };
  return axes[name];
}

/**
 * Campo informativo chave-valor de resultado monoespaçado.
 *
 * @param props.label - Rótulo do campo.
 * @param props.value - Valor textual do campo.
 * @returns Bloco de definição com dt e dd.
 */
function ResultField({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-muted">{label}</dt>
      <dd className="mt-1 break-all font-mono font-medium">{value}</dd>
    </div>
  );
}

/**
 * Card contendo bloco formatado de inspeção JSON de propriedades ou configurações.
 *
 * @param props.id - ID HTML do elemento.
 * @param props.title - Título do card.
 * @param props.value - Objeto a ser serializado em JSON legível.
 * @returns Card contendo visualizador de JSON.
 */
function JsonCard({
  id,
  title,
  value,
}: {
  id: string;
  title: string;
  value: unknown;
}) {
  return (
    <Card id={id}>
      <h2 className="text-lg font-semibold">{title}</h2>
      <pre className="mt-4 max-h-80 overflow-auto rounded-lg bg-code p-4 text-xs text-code-text">
        {JSON.stringify(value, null, 2)}
      </pre>
    </Card>
  );
}
