import { useTranslation } from "react-i18next";

const steps = [
  "dataset",
  "objective",
  "strategy",
  "configuration",
  "training",
  "results",
];

/**
 * Barra de navegação visual indicando as etapas do fluxo de experimentação de Machine Learning.
 *
 * @param props.current - Índice numérico da etapa atualmente em foco (1 a 6).
 * @returns Elemento nav semântico com a lista de etapas numeradas e estilizadas.
 */
export function WorkflowSteps({ current }: { current: number }) {
  const { t } = useTranslation("common");
  return (
    <nav aria-label={t("workflow.label")} className="overflow-x-auto pb-1">
      <ol className="flex min-w-[680px] items-center gap-2">
        {steps.map((step, index) => {
          const number = index + 1;
          const active = number === current;
          const complete = number < current;
          return (
            <li
              key={step}
              aria-current={active ? "step" : undefined}
              className={`flex flex-1 items-center gap-2 rounded-lg border px-3 py-2 text-xs ${
                active
                  ? "border-primary bg-primary text-on-primary"
                  : complete
                    ? "border-success/40 bg-success/10 text-foreground"
                    : "border-border bg-surface text-muted"
              }`}
            >
              <span className="font-mono font-semibold">{number}</span>
              <span className="font-medium">{t(`workflow.steps.${step}`)}</span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
