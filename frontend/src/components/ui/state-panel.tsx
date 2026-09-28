import { Card } from "./card";
import { Button } from "./button";

/**
 * Painel informativo para estados vazios (empty state), carregamento ou erro com ação opcional.
 *
 * @param props.title - Título principal do estado.
 * @param props.description - Descrição complementar opcional.
 * @param props.tone - Tom visual do painel ('neutral' ou 'error'). Padrão 'neutral'.
 * @param props.actionLabel - Rótulo do botão de ação opcional.
 * @param props.onAction - Callback disparado ao clicar no botão de ação.
 * @returns Card contendo mensagem de estado e botão de ação se configurado.
 */
export function StatePanel({
  title,
  description,
  tone = "neutral",
  actionLabel,
  onAction,
}: {
  title: string;
  description?: string;
  tone?: "neutral" | "error";
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <Card
      className="py-10 text-center"
      role={tone === "error" ? "alert" : "status"}
      aria-live="polite"
    >
      <p
        className={
          tone === "error" ? "font-semibold text-error" : "font-semibold"
        }
      >
        {title}
      </p>
      {description ? (
        <p className="mt-2 text-sm text-muted">{description}</p>
      ) : null}
      {actionLabel && onAction ? (
        <Button className="mt-5" variant="outline" onClick={onAction}>
          {actionLabel}
        </Button>
      ) : null}
    </Card>
  );
}
