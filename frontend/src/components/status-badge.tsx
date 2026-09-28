import { useTranslation } from "react-i18next";

const statusTone: Record<string, string> = {
  completed: "border-success/30 bg-success/10 text-success",
  failed: "border-error/30 bg-error/10 text-error",
  running: "border-primary/30 bg-primary/10 text-primary",
  queued: "border-primary/30 bg-primary/10 text-primary",
  cancelled: "border-border bg-background text-muted",
  pruned: "border-warning/30 bg-warning/10 text-warning",
  draft: "border-border bg-background text-foreground",
};

/**
 * Badge colorido indicativo do status de ciclo de vida de experimentos, jobs e modelos.
 *
 * @param props.status - String identificadora do status (ex: completed, failed, running).
 * @returns Elemento span estilizado com cores semânticas e rótulo traduzido.
 */
export function StatusBadge({ status }: { status: string }) {
  const { t } = useTranslation("experiments");
  return (
    <span
      className={`rounded-full border px-2.5 py-1 text-xs font-medium ${
        statusTone[status] ?? statusTone.draft
      }`}
    >
      {t(`status.${status}`, { defaultValue: status })}
    </span>
  );
}
