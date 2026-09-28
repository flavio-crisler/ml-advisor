import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import Plot from "react-plotly.js";
import { useTranslation } from "react-i18next";

import { AIExplanation } from "../../components/ai-explanation";
import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { HelpLabel } from "../../components/ui/info-tooltip";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkspaceLayout } from "../../components/workspace-header";
import {
  createConfidenceCascade,
  listConfidenceCascades,
  type CascadeRun,
} from "../../services/confidence-cascades";
import { listDecisionRuns } from "../../services/decision-runs";
import { DecisionLabNav } from "../decision-providers/decision-lab-nav";
import { HowItWorksButton } from "../../components/how-it-works";

type ThresholdPoint = {
  threshold: number;
  coverage: number;
  auto_decision_rate: number;
  review_rate: number;
  abstain_rate: number;
  accuracy_auto: number | null;
  error_rate_auto: number | null;
  estimated_total_cost: number | null;
};

type CascadeMetrics = ThresholdPoint & {
  decision_cost: number | null;
  threshold_sensitivity: ThresholdPoint[];
  confidence_distribution: Array<{
    lower: number;
    upper: number;
    count: number;
  }>;
};

/**
 * Página do simulador de cascata de confiança para roteamento híbrido (decisão automática vs revisão manual).
 *
 * @returns Interface interativa com sliders de limiares, histogramas de confiança e curvas de sensibilidade.
 */
export function ConfidenceCascadePage() {
  const { t } = useTranslation("decisionLab");
  const queryClient = useQueryClient();
  const decisionRuns = useQuery({
    queryKey: ["decision-runs"],
    queryFn: listDecisionRuns,
  });
  const cascades = useQuery({
    queryKey: ["confidence-cascades"],
    queryFn: listConfidenceCascades,
  });
  const [decisionRunId, setDecisionRunId] = useState("");
  const [autoThreshold, setAutoThreshold] = useState(0.9);
  const [reviewThreshold, setReviewThreshold] = useState(0.6);
  const [action, setAction] = useState<"manual_review" | "abstain">(
    "manual_review",
  );
  const [reviewCost, setReviewCost] = useState("");
  const [selected, setSelected] = useState<CascadeRun | null>(null);
  const create = useMutation({
    mutationFn: createConfidenceCascade,
    onSuccess: (result) => {
      setSelected(result);
      setAutoThreshold(result.policy.auto_accept_threshold);
      queryClient.invalidateQueries({ queryKey: ["confidence-cascades"] });
    },
  });
  const completedRuns =
    decisionRuns.data?.filter((item) => item.status === "completed") ?? [];
  const selectedDecisionRunId = decisionRunId || completedRuns[0]?.id || "";
  const selectedDecisionRun = completedRuns.find(
    (run) => run.id === selectedDecisionRunId,
  );

  function simulate() {
    create.mutate({
      version: "1.0",
      source_decision_run_id: selectedDecisionRunId,
      auto_accept_threshold: autoThreshold,
      review_threshold: reviewThreshold,
      below_review_action: action,
      manual_review_unit_cost: reviewCost ? Number(reviewCost) : null,
    });
  }

  function reopen(run: CascadeRun) {
    setSelected(run);
    setDecisionRunId(run.decision_run_id);
    setAutoThreshold(run.policy.auto_accept_threshold);
    setReviewThreshold(run.policy.review_threshold);
    setAction(run.policy.below_review_action);
    setReviewCost(
      run.policy.manual_review_unit_cost === null ||
        run.policy.manual_review_unit_cost === undefined
        ? ""
        : String(run.policy.manual_review_unit_cost),
    );
  }

  const metrics = selected?.metrics as CascadeMetrics | undefined;
  const selectedPolicy = selected?.policy;
  const current = metrics?.threshold_sensitivity.reduce((closest, point) =>
    Math.abs(point.threshold - autoThreshold) <
    Math.abs(closest.threshold - autoThreshold)
      ? point
      : closest,
  );
  const plotLayout = {
    autosize: true,
    margin: { l: 50, r: 20, t: 20, b: 45 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
  };

  return (
    <WorkspaceLayout>
      <DecisionLabNav />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
            {t("cascade.eyebrow")}
          </p>
          <h1 className="text-4xl font-semibold tracking-tight">
            {t("cascade.title")}
          </h1>
          <p className="mt-3 max-w-3xl text-muted">
            {t("cascade.description")}
          </p>
        </div>
        <HowItWorksButton screen="confidenceCascade" />
      </section>

      <Card>
        <h2 className="text-xl font-semibold">{t("cascade.configure")}</h2>
        <div className="mt-5 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          <Field
            label={t("cascade.sourceRun")}
            help={t("cascade.help.sourceRun")}
          >
            <select
              className="w-full rounded-lg border border-border bg-background px-3 py-2"
              value={selectedDecisionRunId}
              onChange={(event) => setDecisionRunId(event.target.value)}
            >
              <option value="">{t("cascade.selectRun")}</option>
              {completedRuns.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.spec.question.instructions}
                </option>
              ))}
            </select>
          </Field>
          <Field
            label={t("cascade.autoThreshold")}
            help={t("cascade.help.autoThreshold")}
          >
            <input
              className="w-full accent-primary"
              aria-label={t("cascade.autoThreshold")}
              type="range"
              min="0.5"
              max="1"
              step="0.05"
              value={autoThreshold}
              onChange={(event) => setAutoThreshold(Number(event.target.value))}
            />
            <span className="font-mono text-xs">
              {autoThreshold.toFixed(2)}
            </span>
          </Field>
          <Field
            label={t("cascade.reviewThreshold")}
            help={t("cascade.help.reviewThreshold")}
          >
            <input
              className="w-full accent-primary"
              aria-label={t("cascade.reviewThreshold")}
              type="range"
              min="0"
              max={autoThreshold}
              step="0.05"
              value={Math.min(reviewThreshold, autoThreshold)}
              onChange={(event) =>
                setReviewThreshold(Number(event.target.value))
              }
            />
            <span className="font-mono text-xs">
              {Math.min(reviewThreshold, autoThreshold).toFixed(2)}
            </span>
          </Field>
          <Field
            label={t("cascade.lowConfidenceAction")}
            help={t("cascade.help.lowConfidenceAction")}
          >
            <select
              className="w-full rounded-lg border border-border bg-background px-3 py-2"
              value={action}
              onChange={(event) =>
                setAction(event.target.value as "manual_review" | "abstain")
              }
            >
              <option value="manual_review">{t("cascade.manualReview")}</option>
              <option value="abstain">{t("cascade.abstain")}</option>
            </select>
          </Field>
        </div>
        <Field
          label={t("cascade.reviewCost")}
          help={t("cascade.help.reviewCost")}
          className="mt-5 max-w-sm"
        >
          <input
            className="w-full rounded-lg border border-border bg-background px-3 py-2"
            type="number"
            min="0"
            step="0.01"
            value={reviewCost}
            onChange={(event) => setReviewCost(event.target.value)}
            placeholder={t("cascade.optional")}
          />
        </Field>
        <div className="mt-5 flex items-center gap-3">
          <Button
            onClick={simulate}
            disabled={
              !selectedDecisionRunId ||
              reviewThreshold > autoThreshold ||
              create.isPending
            }
          >
            {create.isPending ? t("cascade.simulating") : t("cascade.simulate")}
          </Button>
          {create.isError ? (
            <span className="text-sm text-error">{t("cascade.error")}</span>
          ) : null}
        </div>
      </Card>

      {metrics && current && selectedPolicy ? (
        <section aria-label={t("cascade.results")}>
          <Card>
            <label className="font-semibold" htmlFor="cascade-explorer">
              <HelpLabel help={t("cascade.help.exploreThreshold")}>
                {t("cascade.exploreThreshold")}
              </HelpLabel>
              : {autoThreshold.toFixed(2)}
            </label>
            <input
              id="cascade-explorer"
              className="mt-4 w-full accent-primary"
              type="range"
              min="0.5"
              max="1"
              step="0.025"
              value={autoThreshold}
              onChange={(event) => setAutoThreshold(Number(event.target.value))}
            />
            <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-6">
              <Metric
                label={t("cascade.coverage")}
                help={t("cascade.help.coverage")}
                value={current.coverage}
                percent
              />
              <Metric
                label={t("cascade.accuracy")}
                help={t("cascade.help.accuracy")}
                value={current.accuracy_auto}
                percent
              />
              <Metric
                label={t("cascade.errorRate")}
                help={t("cascade.help.errorRate")}
                value={current.error_rate_auto}
                percent
              />
              <Metric
                label={t("cascade.reviewRate")}
                help={t("cascade.help.reviewRate")}
                value={current.review_rate}
                percent
              />
              <Metric
                label={t("cascade.abstainRate")}
                help={t("cascade.help.abstainRate")}
                value={current.abstain_rate}
                percent
              />
              <Metric
                label={t("cascade.cost")}
                help={t("cascade.help.cost")}
                value={current.estimated_total_cost}
              />
            </div>
            <p className="mt-4 text-xs text-muted">
              {t("cascade.persistedOnly")}
            </p>
          </Card>
          <div className="mt-5">
            <AIExplanation
              scope="confidence_cascade"
              context={{
                source_run: selectedDecisionRun
                  ? {
                      provider_id: selectedDecisionRun.provider_id,
                      requested_model: selectedDecisionRun.requested_model,
                      resolved_model: selectedDecisionRun.resolved_model,
                      question: selectedDecisionRun.spec.question.instructions,
                      row_count: selectedDecisionRun.row_count,
                    }
                  : null,
                policy: selectedPolicy,
                metrics,
                explored_auto_accept_threshold: autoThreshold,
                explored_threshold_metrics: current,
              }}
            />
          </div>
          <div className="mt-5 grid gap-5 xl:grid-cols-2">
            <Chart
              title={t("cascade.thresholdChart")}
              help={t("cascade.help.thresholdChart")}
            >
              <Plot
                data={[
                  {
                    x: metrics.threshold_sensitivity.map(
                      (item) => item.threshold,
                    ),
                    y: metrics.threshold_sensitivity.map(
                      (item) => item.coverage,
                    ),
                    type: "scatter",
                    mode: "lines+markers",
                    name: t("cascade.coverage"),
                  },
                  {
                    x: metrics.threshold_sensitivity.map(
                      (item) => item.threshold,
                    ),
                    y: metrics.threshold_sensitivity.map(
                      (item) => item.accuracy_auto,
                    ),
                    type: "scatter",
                    mode: "lines+markers",
                    name: t("cascade.accuracy"),
                  },
                  {
                    x: metrics.threshold_sensitivity.map(
                      (item) => item.threshold,
                    ),
                    y: metrics.threshold_sensitivity.map(
                      (item) => item.review_rate,
                    ),
                    type: "scatter",
                    mode: "lines+markers",
                    name: t("cascade.reviewRate"),
                  },
                  {
                    x: metrics.threshold_sensitivity.map(
                      (item) => item.threshold,
                    ),
                    y: metrics.threshold_sensitivity.map(
                      (item) => item.abstain_rate,
                    ),
                    type: "scatter",
                    mode: "lines+markers",
                    name: t("cascade.abstainRate"),
                  },
                  {
                    x: metrics.threshold_sensitivity.map(
                      (item) => item.threshold,
                    ),
                    y: metrics.threshold_sensitivity.map(
                      (item) => item.error_rate_auto,
                    ),
                    type: "scatter",
                    mode: "lines+markers",
                    name: t("cascade.errorRate"),
                  },
                  ...(metrics.threshold_sensitivity.some(
                    (item) => item.estimated_total_cost !== null,
                  )
                    ? [
                        {
                          x: metrics.threshold_sensitivity.map(
                            (item) => item.threshold,
                          ),
                          y: metrics.threshold_sensitivity.map(
                            (item) => item.estimated_total_cost,
                          ),
                          type: "scatter" as const,
                          mode: "lines+markers" as const,
                          name: t("cascade.cost"),
                          yaxis: "y2" as const,
                        },
                      ]
                    : []),
                ]}
                layout={{
                  ...plotLayout,
                  xaxis: { title: { text: t("cascade.thresholdAxis") } },
                  yaxis: {
                    title: { text: t("cascade.rateAxis") },
                    range: [0, 1],
                  },
                  yaxis2: {
                    overlaying: "y",
                    side: "right",
                    showgrid: false,
                    title: { text: t("cascade.costAxis") },
                  },
                }}
                useResizeHandler
                className="h-72 w-full"
                config={{ displayModeBar: false, responsive: true }}
              />
            </Chart>
            <Chart
              title={t("cascade.distribution")}
              help={t("cascade.help.distribution")}
            >
              <Plot
                data={[
                  {
                    x: metrics.confidence_distribution.map(
                      (item) => (item.lower + item.upper) / 2,
                    ),
                    y: metrics.confidence_distribution.map(
                      (item) => item.count,
                    ),
                    type: "bar",
                  },
                ]}
                layout={{
                  ...plotLayout,
                  xaxis: { title: { text: t("cascade.confidenceAxis") } },
                  yaxis: { title: { text: t("cascade.sampleCountAxis") } },
                }}
                useResizeHandler
                className="h-72 w-full"
                config={{ displayModeBar: false, responsive: true }}
              />
            </Chart>
          </div>
        </section>
      ) : null}

      <section>
        <h2 className="mb-4 text-xl font-semibold">{t("cascade.history")}</h2>
        {cascades.isPending ? (
          <StatePanel title={t("states.loading")} />
        ) : cascades.isError ? (
          <StatePanel
            title={t("cascade.historyError")}
            description={t("states.errorDescription")}
            tone="error"
            actionLabel={t("states.retry")}
            onAction={() => void cascades.refetch()}
          />
        ) : cascades.data?.length ? (
          <div className="grid gap-3 md:grid-cols-2">
            {cascades.data.map((run) => (
              <button
                key={run.id}
                className="text-left"
                onClick={() => reopen(run)}
              >
                <Card className="h-full py-4 transition-colors hover:border-primary">
                  <p className="font-mono text-xs text-muted">{run.id}</p>
                  <p className="mt-2 font-semibold">
                    {t("cascade.savedPolicy", {
                      auto: run.policy.auto_accept_threshold.toFixed(2),
                      review: run.policy.review_threshold.toFixed(2),
                    })}
                  </p>
                </Card>
              </button>
            ))}
          </div>
        ) : (
          <StatePanel
            title={t("cascade.empty")}
            description={t("cascade.emptyDescription")}
          />
        )}
      </section>
    </WorkspaceLayout>
  );
}

/**
 * Campo de formulário rotulado com tooltip de suporte integrado.
 *
 * @param props.label - Rótulo do campo.
 * @param props.help - Conteúdo de ajuda contextual.
 * @param props.children - Elemento de entrada ou controle do formulário.
 * @param props.className - Classes CSS adicionais para o contêiner do label.
 * @returns Elemento label estruturado.
 */
function Field({
  label,
  help,
  children,
  className = "",
}: {
  label: string;
  help: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={`block text-sm font-medium ${className}`}>
      <span className="mb-2 block">
        <HelpLabel help={help}>{label}</HelpLabel>
      </span>
      {children}
    </label>
  );
}

/**
 * Indicador de métrica com valor destacado e tooltip de auxílio.
 *
 * @param props.label - Nome da métrica.
 * @param props.help - Explicação do significado da métrica.
 * @param props.value - Valor numérico a ser exibido.
 * @param props.percent - Se o valor deve ser formatado em porcentagem. Padrão false.
 * @returns Bloco informativo da métrica.
 */
function Metric({
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
        {value === null
          ? "—"
          : percent
            ? `${(value * 100).toFixed(1)}%`
            : value.toFixed(4)}
      </p>
    </div>
  );
}

/**
 * Card de gráfico com cabeçalho contendo tooltip explicativo.
 *
 * @param props.title - Título do gráfico.
 * @param props.help - Conteúdo explicativo contextual.
 * @param props.children - Componente do gráfico Plotly ou visualização equivalente.
 * @returns Card contendo o título e a visualização gráfica.
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
    <Card>
      <h2 className="font-semibold">
        <HelpLabel help={help}>{title}</HelpLabel>
      </h2>
      <div className="mt-3">{children}</div>
    </Card>
  );
}
