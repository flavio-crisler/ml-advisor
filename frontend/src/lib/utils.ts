import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combina classes CSS utilitárias usando clsx e tailwind-merge.
 *
 * @param inputs - Lista de classes CSS, condições ou objetos de classe.
 * @returns String contendo as classes combinadas sem conflitos do Tailwind.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
