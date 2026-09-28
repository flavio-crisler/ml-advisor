import { cn } from "../../lib/utils";

type ProgressProps = {
  value: number;
  className?: string;
};

/**
 * Barra de progresso horizontal acessível com animação de transição suave.
 *
 * @param props.value - Valor percentual atual do progresso (0 a 100).
 * @param props.className - Classes CSS adicionais para customização do container.
 * @returns Elemento div estilizado como barra de progresso.
 */
export function Progress({ value, className }: ProgressProps) {
  const normalizedValue = Math.min(100, Math.max(0, value));

  return (
    <div
      className={cn("h-2 overflow-hidden rounded-full bg-border", className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={normalizedValue}
    >
      <div
        className="h-full rounded-full bg-primary transition-[width]"
        style={{ width: `${normalizedValue}%` }}
      />
    </div>
  );
}
