import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import {
  explainResult,
  getExplanationAvailability,
  type ExplanationLanguage,
  type ExplanationScope,
  type ResultExplanation,
} from "../services/explanations";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import {
  AIImprovementProposal,
  type ImprovementSource,
} from "./ai-improvement-proposal";

/**
 * Componente interativo para geração sob demanda de explicações via IA sobre execuções e métricas.
 *
 * @param props.scope - Escopo da explicação (ex: experiment_comparison, run_analysis).
 * @param props.context - Dados contextuais agregados do experimento ou gráfico.
 * @param props.improvementSource - Fonte opcional de propostas de melhoria iterativa associada.
 * @param props.onExplanationChange - Callback opcional disparado quando uma explicação é gerada ou atualizada.
 * @returns Card contendo botão de acionamento, status de disponibilidade e relatório explicativo.
 */
export function AIExplanation({
  scope,
  context,
  improvementSource,
  onExplanationChange,
}: {
  scope: ExplanationScope;
  context: Record<string, unknown>;
  improvementSource?: ImprovementSource;
  onExplanationChange?: (explanation: ResultExplanation | undefined) => void;
}) {
  const { t, i18n } = useTranslation("common");
  const language: ExplanationLanguage = i18n.resolvedLanguage?.startsWith("pt")
    ? "pt-BR"
    : "en-US";
  const contextFingerprint = JSON.stringify(context);
  const availability = useQuery({
    queryKey: ["explanations", "status"],
    queryFn: getExplanationAvailability,
    staleTime: 60_000,
    retry: false,
  });
  const explanation = useMutation({
    mutationFn: explainResult,
  });
  const resultMatchesContext =
    explanation.variables?.scope === scope &&
    explanation.variables.language === language &&
    JSON.stringify(explanation.variables.context) === contextFingerprint;
  const activeResult = resultMatchesContext ? explanation.data : undefined;

  useEffect(() => {
    onExplanationChange?.(activeResult?.explanation);
  }, [activeResult?.explanation, onExplanationChange]);

  const provider = availability.data;
  const available = provider?.available === true;

  return (
    <Card aria-labelledby={`ai-explanation-${scope}`}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-3xl">
          <h2 id={`ai-explanation-${scope}`} className="text-xl font-semibold">
            {t("aiExplanation.title")}
          </h2>
          <p className="mt-2 text-sm text-muted">
            {t("aiExplanation.description")}
          </p>
          {provider ? (
            <p className="mt-2 text-xs text-muted">
              {t(
                provider.is_local
                  ? "aiExplanation.localNotice"
                  : "aiExplanation.externalNotice",
                { provider: provider.provider_id },
              )}
            </p>
          ) : null}
        </div>
        <Button
          onClick={() => explanation.mutate({ scope, language, context })}
          disabled={
            availability.isPending ||
            availability.isError ||
            !available ||
            explanation.isPending
          }
        >
          {explanation.isPending
            ? t("aiExplanation.loading")
            : activeResult
              ? t("aiExplanation.regenerate")
              : t("aiExplanation.action")}
        </Button>
      </div>

      {!availability.isPending && !available ? (
        <p className="mt-4 text-sm text-warning">
          {t("aiExplanation.unavailable")}
        </p>
      ) : null}
      {explanation.isError && resultMatchesContext ? (
        <p className="mt-4 text-sm text-error">{t("aiExplanation.error")}</p>
      ) : null}
      {activeResult ? (
        <div className="mt-6 border-t border-border pt-5" aria-live="polite">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
            {t("aiExplanation.generatedLabel")}
          </p>
          <p className="mt-3 leading-7">{activeResult.explanation.summary}</p>
          <div className="mt-5 grid gap-5 lg:grid-cols-3">
            <ExplanationList
              title={t("aiExplanation.findings")}
              items={activeResult.explanation.findings ?? []}
            />
            <ExplanationList
              title={t("aiExplanation.cautions")}
              items={activeResult.explanation.cautions ?? []}
            />
            <ExplanationList
              title={t("aiExplanation.nextSteps")}
              items={activeResult.explanation.next_steps ?? []}
            />
          </div>
          <p className="mt-5 text-xs text-muted">
            {t("aiExplanation.disclaimer")}
          </p>
          {improvementSource ? (
            <AIImprovementProposal
              source={improvementSource}
              language={language}
              context={context}
            />
          ) : null}
        </div>
      ) : null}
    </Card>
  );
}

/**
 * Renderiza uma seção com título e lista de tópicos explicativos (achados, cuidados ou próximos passos).
 *
 * @param props.title - Título do bloco.
 * @param props.items - Vetor de strings com os itens da lista.
 * @returns Elemento de seção com lista HTML ou null se vazia.
 */
function ExplanationList({
  title,
  items,
}: {
  title: ReactNode;
  items: string[];
}) {
  if (!items.length) return null;
  return (
    <section>
      <h3 className="font-semibold">{title}</h3>
      <ul className="mt-2 list-disc space-y-2 pl-5 text-sm text-muted">
        {items.map((item, index) => (
          <li key={`${index}-${item}`}>{item}</li>
        ))}
      </ul>
    </section>
  );
}
