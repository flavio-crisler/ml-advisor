import { useMutation, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";

import { Button } from "../../components/ui/button";
import { getExplanationAvailability } from "../../services/explanations";
import { explainSemanticImpact } from "../../services/semantic-features";

/**
 * Componente que aciona e renderiza a explicação textual por IA sobre o impacto das features semânticas no modelo.
 *
 * @param props.featureSetId - Identificador único do conjunto de features semânticas avaliado.
 * @returns Seção com botão de geração, avisos de indisponibilidade e resumo de achados e próximos passos.
 */
export function SemanticImpactAI({ featureSetId }: { featureSetId: string }) {
  const { t, i18n } = useTranslation("decisionLab");
  const language = i18n.resolvedLanguage?.startsWith("pt") ? "pt-BR" : "en-US";
  const availability = useQuery({
    queryKey: ["explanations", "status"],
    queryFn: getExplanationAvailability,
    staleTime: 60_000,
    retry: false,
  });
  const explanation = useMutation({
    mutationFn: (requestedLanguage: "pt-BR" | "en-US") =>
      explainSemanticImpact(featureSetId, requestedLanguage),
  });
  const result =
    explanation.variables === language ? explanation.data : undefined;

  return (
    <section className="mt-5">
      <Button
        title={
          !availability.isPending && availability.data?.available !== true
            ? t("semantic.aiUnavailable")
            : undefined
        }
        disabled={
          availability.data?.available !== true || explanation.isPending
        }
        onClick={() => explanation.mutate(language)}
      >
        {explanation.isPending
          ? t("semantic.aiLoading")
          : t("semantic.aiAction")}
      </Button>
      {explanation.isError ? (
        <p className="mt-2 text-sm text-error" role="alert">
          {t("semantic.aiError")}
        </p>
      ) : null}
      {result ? (
        <div className="mt-4 border-t border-border pt-4" aria-live="polite">
          <p>{result.explanation.summary}</p>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {(
              [
                ["aiFindings", result.explanation.findings],
                ["aiCautions", result.explanation.cautions],
                ["aiNextSteps", result.explanation.next_steps],
              ] as const
            ).map(([title, items]) => (
              <div key={title}>
                <h5 className="font-semibold">{t(`semantic.${title}`)}</h5>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted">
                  {(items ?? []).map((item, index) => (
                    <li key={`${title}-${index}`}>{item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
