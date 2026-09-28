import { useMutation, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import {
  explainResult,
  getExplanationAvailability,
  type ChartExplanation,
  type ExplanationLanguage,
} from "../services/explanations";
import { Button } from "./ui/button";

/**
 * Componente que renderiza a explicação focada de um gráfico específico por meio de IA generativa.
 *
 * @param props.chartId - Identificador do gráfico na análise (ex: confusion_matrix, shap_summary).
 * @param props.context - Contexto agregado do gráfico para envio ao modelo de linguagem.
 * @param props.globalExplanation - Explicação pré-existente opcional herdada da análise geral.
 * @returns Seção colapsável ou expandida com a interpretação e achados do gráfico.
 */
export function AIChartExplanation({
  chartId,
  context,
  globalExplanation,
}: {
  chartId: string;
  context: Record<string, unknown>;
  globalExplanation?: ChartExplanation;
}) {
  const { t, i18n } = useTranslation("common");
  const language: ExplanationLanguage = i18n.resolvedLanguage?.startsWith("pt")
    ? "pt-BR"
    : "en-US";
  const availability = useQuery({
    queryKey: ["explanations", "status"],
    queryFn: getExplanationAvailability,
    staleTime: 60_000,
    retry: false,
  });
  const explanation = useMutation({ mutationFn: explainResult });
  const contextFingerprint = JSON.stringify(context);
  const resultMatchesContext =
    explanation.variables?.scope === "chart_explanation" &&
    explanation.variables.language === language &&
    JSON.stringify(explanation.variables.context) === contextFingerprint;
  const focusedExplanation = resultMatchesContext
    ? explanation.data?.explanation.chart_explanations?.find(
        (item) => item.chart_id === chartId,
      )
    : undefined;
  const displayedExplanation = focusedExplanation ?? globalExplanation;
  const available = availability.data?.available === true;

  return (
    <section className="mt-4 border-t border-border pt-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold">{t("aiChart.title")}</p>
          <p className="mt-1 text-xs text-muted">{t("aiChart.description")}</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          disabled={
            availability.isPending ||
            availability.isError ||
            !available ||
            explanation.isPending
          }
          onClick={() =>
            explanation.mutate({
              scope: "chart_explanation",
              language,
              context,
            })
          }
        >
          {explanation.isPending
            ? t("aiChart.loading")
            : focusedExplanation
              ? t("aiChart.regenerate")
              : t("aiChart.action")}
        </Button>
      </div>

      {!availability.isPending && !available ? (
        <p className="mt-3 text-xs text-warning">
          {t("aiExplanation.unavailable")}
        </p>
      ) : null}
      {explanation.isError && resultMatchesContext ? (
        <p className="mt-3 text-xs text-error">{t("aiChart.error")}</p>
      ) : null}
      {displayedExplanation ? (
        <div className="mt-4 rounded-lg bg-primary/5 p-4" aria-live="polite">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-primary">
            {focusedExplanation
              ? t("aiChart.focusedLabel")
              : t("aiChart.generatedLabel")}
          </p>
          <p className="mt-2 text-sm leading-6">
            {displayedExplanation.summary}
          </p>
          <ChartList
            title={t("aiExplanation.findings")}
            items={displayedExplanation.findings ?? []}
          />
          <ChartList
            title={t("aiExplanation.cautions")}
            items={displayedExplanation.cautions ?? []}
          />
        </div>
      ) : null}
    </section>
  );
}

/**
 * Lista de tópicos específicos de um gráfico individual (achados ou cuidados).
 *
 * @param props.title - Título do conjunto de tópicos.
 * @param props.items - Itens de texto a listar.
 * @returns Lista de itens ou null se a lista estiver vazia.
 */
function ChartList({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div className="mt-3">
      <p className="text-xs font-semibold">{title}</p>
      <ul className="mt-1 list-disc space-y-1 pl-5 text-xs text-muted">
        {items.map((item, index) => (
          <li key={`${index}-${item}`}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
