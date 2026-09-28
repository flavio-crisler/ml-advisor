import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "../ui/button";
import {
  HowItWorksModal,
  type HowItWorksScreenKey,
} from "./how-it-works-modal";

/**
 * Propriedades do botão de ajuda interativo "Como Funciona".
 */
interface HowItWorksButtonProps {
  screen: HowItWorksScreenKey;
  className?: string;
  variant?: "default" | "outline";
  size?: "default" | "sm";
}

/**
 * Botão que aciona a modal com a documentação conceitual e instruções da tela atual.
 *
 * @param props.screen - Chave identificadora da tela atual para carregar o conteúdo correto.
 * @param props.className - Classes CSS adicionais.
 * @param props.variant - Variante visual do botão. Padrão 'outline'.
 * @param props.size - Tamanho do botão. Padrão 'sm'.
 * @returns Elemento de botão e modal correspondente acoplada.
 */
export function HowItWorksButton({
  screen,
  className = "",
  variant = "outline",
  size = "sm",
}: HowItWorksButtonProps) {
  const { t } = useTranslation("howItWorks");
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <Button
        type="button"
        variant={variant}
        size={size}
        className={`inline-flex items-center gap-1.5 font-medium ${className}`}
        onClick={() => setIsOpen(true)}
      >
        <svg
          className="h-4 w-4 text-primary"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span>{t("buttonLabel")}</span>
      </Button>
      <HowItWorksModal
        screen={screen}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
      />
    </>
  );
}
