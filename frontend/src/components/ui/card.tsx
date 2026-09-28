import type { HTMLAttributes } from "react";

import { cn } from "../../lib/utils";

/**
 * Componente de container estilo card com borda, sombreamento e superfície padronizados.
 *
 * @param props - Propriedades padrão de um elemento HTML div.
 * @returns Elemento div estilizado como cartão.
 */
export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-surface p-6 shadow-card",
        className,
      )}
      {...props}
    />
  );
}
