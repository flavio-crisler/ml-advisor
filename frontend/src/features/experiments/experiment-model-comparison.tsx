import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { StatusBadge } from "../../components/status-badge";
import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import {
  getExperimentComparison,
  explainExperimentComparison,
  type ExperimentComparison,
} from "../../services/experiments";
import {
  getExplanationAvailability,
  type ExplanationLanguage,
  type ResultExplanation,
} from "../../services/explanations";

/**
 * Propriedades para renderização do comparativo de modelos do experimento.
 */
interface ExperimentModelComparisonProps {
  experimentId: string;
}

/**
 * Componente que exibe a tabela ranqueada de modelos, modelo campeão, exclusões e explicação via IA.
 *
 * @param props.experimentId - Identificador único do experimento.
 * @returns Painel com ranking de modelos avaliados por rodada e interpretação executiva.
 */
export function ExperimentModelComparison({
  experimentId,
}: ExperimentModelComparisonProps) {
  const { t, i18n } = useTranslation("experiments");
  const language: ExplanationLanguage = i18n.resolvedLanguage?.startsWith("pt")
    ? "pt-BR"
    : "en-US";

  const [selectedRoundId, setSelectedRoundId] = useState<string | null>(null);
  const [showExcluded, setShowExcluded] = useState(false);

  const availabilityQuery = useQuery({
    queryKey: ["explanations", "status"],
    queryFn: getExplanationAvailability,
    staleTime: 60_000,
    retry: false,
  });

  const comparisonQuery = useQuery<ExperimentComparison>({
    queryKey: ["experiment-model-comparison", experimentId, selectedRoundId],
    queryFn: () =>
      getExperimentComparison(experimentId, selectedRoundId ?? undefined),
    enabled: Boolean(experimentId),
  });

  const comparison = comparisonQuery.data;
  const effectiveRoundId = selectedRoundId ?? comparison?.selected_round_id;

  const explanationMutation = useMutation({
    mutationFn: (vars: {
      executionId: string;
      language: ExplanationLanguage;
    }) =>
      explainExperimentComparison(
        experimentId,
        vars.executionId,
        vars.language,
      ),
  });

  const { reset: resetExplanation } = explanationMutation;

  // Reset or invalidate explanation when round or language changes
  useEffect(() => {
    resetExplanation();
  }, [effectiveRoundId, language, resetExplanation]);

  const provider = availabilityQuery.data;
  const isAiAvailable = provider?.available === true;

  const resultMatchesContext =
    explanationMutation.variables?.executionId === effectiveRoundId &&
    explanationMutation.variables?.language === language;
  const activeResult = resultMatchesContext
    ? explanationMutation.data
    : undefined;
  const explanationData = activeResult?.explanation as
    ResultExplanation | undefined;

  const rankedModels = Array.isArray(comparison?.ranked_models)
    ? comparison.ranked_models
    : [];
  const excludedModels = Array.isArray(comparison?.excluded_models)
    ? comparison.excluded_models
    : [];
  const availableRounds = Array.isArray(comparison?.available_rounds)
    ? comparison.available_rounds
    : [];
  const canExplain = Boolean(comparison?.can_explain);

  const formatDelta = (delta: number) => {
    if (delta === 0) return "—";
    return delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4);
  };

  const formatDuration = (seconds?: number | null) => {
    if (seconds == null) return "—";
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const rem = (seconds % 60).toFixed(0);
    return `${mins}m ${rem}s`;
  };

  const formatRoundLabel = (
    round: NonNullable<typeof comparison>["available_rounds"][number],
    index: number,
  ) => {
    if (!round) return "";
    const isLegacy = round.execution_id.startsWith("legacy");
    if (isLegacy) {
      return t("modelComparison.roundOptionLegacy", {
        completed: round.completed_models ?? 0,
        total: round.total_models ?? 0,
      });
    }
    const dateStr = round.started_at
      ? new Date(round.started_at).toLocaleDateString(
          language === "pt-BR" ? "pt-BR" : "en-US",
          {
            day: "2-digit",
            month: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
          },
        )
      : "";
    if ((round.completed_models ?? 0) < (round.total_models ?? 0)) {
      return t("modelComparison.roundOptionRunning", {
        index,
        date: dateStr,
        completed: round.completed_models ?? 0,
        total: round.total_models ?? 0,
      });
    }
    return t("modelComparison.roundOption", {
      index,
      date: dateStr,
      completed: round.completed_models ?? 0,
      total: round.total_models ?? 0,
    });
  };

  return (
    <Card
      className="mt-8 overflow-hidden"
      aria-labelledby="model-comparison-heading"
      data-testid="model-comparison-panel"
    >
      {/* Header section with title, round selector and metric direction */}
      <div className="flex flex-col gap-4 border-b border-border pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2
            id="model-comparison-heading"
            className="text-xl font-semibold text-foreground"
          >
            {t("modelComparison.title")}
          </h2>
          <p className="mt-1 text-sm text-muted">
            {t("modelComparison.description")}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {comparison?.primary_metric ? (
            <div className="flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
              <span>{comparison.primary_metric}</span>
              <span className="text-muted/60">•</span>
              <span>
                {comparison.metric_direction === "maximize"
                  ? t("modelComparison.directionMaximize")
                  : t("modelComparison.directionMinimize")}
              </span>
            </div>
          ) : null}

          {availableRounds.length > 1 ? (
            <div className="flex items-center gap-2">
              <label
                htmlFor="comparison-round-select"
                className="text-xs font-medium text-muted"
              >
                {t("modelComparison.roundLabel")}:
              </label>
              <select
                id="comparison-round-select"
                data-testid="comparison-round-select"
                className="rounded-md border border-border bg-background px-2.5 py-1 text-xs font-medium text-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                value={effectiveRoundId ?? ""}
                onChange={(e) => {
                  setSelectedRoundId(e.target.value);
                  setShowExcluded(false);
                }}
              >
                {availableRounds.map((r, i) => (
                  <option key={r.execution_id} value={r.execution_id}>
                    {formatRoundLabel(r, i + 1)}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </div>
      </div>

      {/* Main comparison table or states */}
      <div className="mt-5">
        {comparisonQuery.isPending ? (
          <div className="py-8 text-center text-sm text-muted">
            {t("loading")}
          </div>
        ) : comparisonQuery.isError ? (
          <div className="rounded-lg border border-error/30 bg-error/10 p-4 text-sm text-error">
            {t("errors.load")}
          </div>
        ) : rankedModels.length === 0 ? (
          <div
            className="rounded-lg border border-border bg-muted/20 p-6 text-center text-sm text-muted"
            data-testid="no-models-state"
          >
            {t("modelComparison.noModels")}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table
              className="w-full text-left text-sm"
              aria-label={t("modelComparison.title")}
            >
              <thead>
                <tr className="border-b border-border text-xs font-semibold text-muted">
                  <th scope="col" className="pb-3 pr-4 text-center w-14">
                    {t("modelComparison.rank")}
                  </th>
                  <th scope="col" className="pb-3 pr-4">
                    {t("modelComparison.model")}
                  </th>
                  <th scope="col" className="pb-3 pr-4 text-right">
                    {t("modelComparison.primaryMetric", {
                      metric: comparison?.primary_metric,
                    })}
                  </th>
                  <th scope="col" className="pb-3 pr-4 text-right">
                    {t("modelComparison.deltaToBest")}
                  </th>
                  <th scope="col" className="pb-3 pr-4 text-right">
                    {t("modelComparison.foldStdDev")}
                  </th>
                  <th scope="col" className="pb-3 pr-4 text-right">
                    {t("modelComparison.duration")}
                  </th>
                  <th scope="col" className="pb-3 text-right">
                    {t("modelComparison.secondaryMetrics")}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60">
                {rankedModels.map((model) => {
                  const isBest = model.rank === 1;
                  return (
                    <tr
                      key={model.run_id}
                      className={`hover:bg-muted/10 transition-colors ${
                        isBest ? "bg-primary/5 font-medium" : ""
                      }`}
                      data-testid={`comparison-row-${model.model_id}`}
                    >
                      <td className="py-3 pr-4 text-center">
                        <span
                          className={`inline-flex items-center justify-center rounded-full text-xs font-bold ${
                            isBest
                              ? "h-6 w-6 bg-primary text-primary-foreground shadow-sm"
                              : "h-5 w-5 text-muted"
                          }`}
                        >
                          {model.rank}
                        </span>
                      </td>
                      <td className="py-3 pr-4">
                        <div className="flex items-center gap-2">
                          <Link
                            to={`/experiments/${experimentId}/runs/${model.run_id}`}
                            className="font-semibold text-primary hover:underline"
                          >
                            {model.model_id}
                          </Link>
                          {isBest ? (
                            <span className="rounded bg-primary/20 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-primary">
                              {t("modelComparison.bestBadge")}
                            </span>
                          ) : null}
                        </div>
                      </td>
                      <td className="py-3 pr-4 text-right font-mono font-bold">
                        {typeof model.primary_metric_value === "number"
                          ? model.primary_metric_value.toFixed(4)
                          : String(model.primary_metric_value)}
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-xs text-muted">
                        {formatDelta(model.delta_to_best)}
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-xs text-muted">
                        {model.standard_deviation != null
                          ? `±${model.standard_deviation.toFixed(4)}`
                          : "—"}
                      </td>
                      <td className="py-3 pr-4 text-right font-mono text-xs text-muted">
                        {formatDuration(model.duration_seconds)}
                      </td>
                      <td className="py-3 text-right">
                        {Object.entries(model.secondary_metrics).length > 0 ? (
                          <div className="flex flex-wrap justify-end gap-2 text-xs font-mono">
                            {Object.entries(model.secondary_metrics).map(
                              ([metricName, metricVal]) => (
                                <span key={metricName} className="text-muted">
                                  <span className="font-sans text-[11px] opacity-75">
                                    {metricName}:
                                  </span>{" "}
                                  {typeof metricVal === "number"
                                    ? metricVal.toFixed(4)
                                    : String(metricVal)}
                                </span>
                              ),
                            )}
                          </div>
                        ) : (
                          <span className="text-xs text-muted">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* State banner when fewer than 2 models are eligible */}
        {rankedModels.length === 1 ? (
          <div
            className="mt-4 rounded-lg border border-warning/30 bg-warning/10 p-3 text-xs text-warning"
            data-testid="insufficient-models-banner"
          >
            {t("modelComparison.insufficientModels")}
          </div>
        ) : null}

        {/* Excluded models collapsible section */}
        {excludedModels.length > 0 ? (
          <div className="mt-4 border-t border-border pt-3">
            <button
              type="button"
              onClick={() => setShowExcluded(!showExcluded)}
              className="flex items-center gap-2 text-xs font-semibold text-muted hover:text-foreground"
              aria-expanded={showExcluded}
            >
              <span>
                {t("modelComparison.excludedTitle", {
                  count: excludedModels.length,
                })}
              </span>
              <span className="text-[10px]">{showExcluded ? "▲" : "▼"}</span>
            </button>
            {showExcluded ? (
              <div className="mt-3 rounded-lg border border-border bg-muted/10 p-3">
                <p className="mb-2 text-xs text-muted">
                  {t("modelComparison.excludedHelp")}
                </p>
                <div className="space-y-2">
                  {excludedModels.map((excluded) => (
                    <div
                      key={excluded.run_id}
                      className="flex items-center justify-between text-xs"
                      data-testid={`excluded-row-${excluded.model_id}`}
                    >
                      <div className="flex items-center gap-2">
                        <Link
                          to={`/experiments/${experimentId}/runs/${excluded.run_id}`}
                          className="font-medium hover:underline text-foreground"
                        >
                          {excluded.model_id}
                        </Link>
                        <StatusBadge status={excluded.status} />
                      </div>
                      <span className="text-muted">
                        {t(`modelComparison.reason.${excluded.reason}`, {
                          defaultValue: excluded.reason,
                        })}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {/* AI Comparative Explanation Section */}
      <div
        className="mt-6 border-t border-border pt-5"
        data-testid="ai-comparison-section"
      >
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-2xl">
            <h3 className="text-base font-semibold text-foreground">
              {t("modelComparison.ai.title")}
            </h3>
            <p className="mt-1 text-sm text-muted">
              {t("modelComparison.ai.description")}
            </p>
            {provider ? (
              <p className="mt-2 text-xs text-muted">
                {t(
                  provider.is_local
                    ? "modelComparison.ai.noticeLocal"
                    : "modelComparison.ai.notice",
                  { provider: provider.provider_id },
                )}
              </p>
            ) : null}
          </div>

          <Button
            type="button"
            data-testid="explain-comparison-button"
            onClick={() => {
              if (effectiveRoundId) {
                explanationMutation.mutate({
                  executionId: effectiveRoundId,
                  language,
                });
              }
            }}
            disabled={
              !canExplain ||
              explanationMutation.isPending ||
              availabilityQuery.isPending ||
              !isAiAvailable
            }
          >
            {explanationMutation.isPending
              ? t("modelComparison.ai.loading")
              : activeResult
                ? t("modelComparison.ai.regenerate")
                : t("modelComparison.ai.action")}
          </Button>
        </div>

        {/* AI explanation loading state */}
        {explanationMutation.isPending ? (
          <div
            className="mt-5 rounded-lg border border-primary/20 bg-primary/5 p-6 text-center text-sm text-primary"
            aria-live="polite"
            data-testid="ai-comparison-loading"
          >
            <div className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent mr-2 align-middle" />
            {t("modelComparison.ai.loading")}
          </div>
        ) : null}

        {/* AI explanation error state */}
        {explanationMutation.isError && resultMatchesContext ? (
          <div
            className="mt-4 rounded-lg border border-error/30 bg-error/10 p-4 text-sm text-error"
            data-testid="ai-comparison-error"
          >
            <p className="font-semibold">{t("modelComparison.ai.error")}</p>
            <div className="mt-3">
              <Button
                type="button"
                variant="outline"
                className="text-xs"
                onClick={() => {
                  if (effectiveRoundId) {
                    explanationMutation.mutate({
                      executionId: effectiveRoundId,
                      language,
                    });
                  }
                }}
              >
                {t("modelComparison.ai.retry")}
              </Button>
            </div>
          </div>
        ) : null}

        {/* AI explanation results display */}
        {explanationData && resultMatchesContext ? (
          <div
            className="mt-5 space-y-4 rounded-lg border border-border bg-card p-5"
            aria-live="polite"
            data-testid="ai-comparison-result"
          >
            {/* Executive summary */}
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wider text-muted">
                {t("modelComparison.ai.summary")}
              </h4>
              <p className="mt-1.5 text-sm leading-relaxed text-foreground">
                {explanationData.summary}
              </p>
            </div>

            {/* Findings / Key takeaways */}
            {explanationData.findings?.length ? (
              <div className="border-t border-border/60 pt-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted">
                  {t("modelComparison.ai.keyTakeaways")}
                </h4>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-foreground">
                  {explanationData.findings.map((item: string, idx: number) => (
                    <li key={idx} className="leading-relaxed">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {/* Next steps */}
            {explanationData.next_steps?.length ? (
              <div className="border-t border-border/60 pt-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted">
                  {t("modelComparison.ai.nextSteps")}
                </h4>
                <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-foreground">
                  {explanationData.next_steps.map(
                    (item: string, idx: number) => (
                      <li key={idx} className="leading-relaxed">
                        {item}
                      </li>
                    ),
                  )}
                </ol>
              </div>
            ) : null}

            {/* Cautions / Limitations */}
            {explanationData.cautions?.length ? (
              <div className="rounded-md border border-warning/30 bg-warning/10 p-3 text-xs text-warning">
                <h4 className="font-semibold uppercase tracking-wider">
                  {t("modelComparison.ai.limitations")}
                </h4>
                <ul className="mt-1 list-disc space-y-1 pl-4">
                  {explanationData.cautions.map((item: string, idx: number) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            {/* Provider footer */}
            {activeResult ? (
              <div className="border-t border-border/60 pt-2 text-right text-[11px] text-muted">
                {t("modelComparison.ai.modelFooter", {
                  provider: activeResult.provider_id,
                })}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </Card>
  );
}
