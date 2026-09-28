import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { Button } from "../ui/button";
import { Card } from "../ui/card";

/**
 * Chaves identificadoras das telas suportadas pelo guia explicativo "Como Funciona".
 */
export type HowItWorksScreenKey =
  | "datasets"
  | "expert"
  | "experiments"
  | "experimentDetail"
  | "runDetail"
  | "decisionProviders"
  | "decisionBenchmark"
  | "calibrationLab"
  | "semanticFeatures"
  | "confidenceCascade";

/**
 * Propriedades da janela modal "Como Funciona".
 */
interface HowItWorksModalProps {
  isOpen: boolean;
  onClose: () => void;
  screen: HowItWorksScreenKey;
}

/**
 * Modal explicativa acessível com abas de Visão Geral, Detalhes Técnicos e Passo a Passo.
 *
 * @param props.isOpen - Se a modal está visível no momento.
 * @param props.onClose - Callback invocado ao fechar a modal (botão fechar, tecla Esc ou backdrop).
 * @param props.screen - Tela atualmente ativa para seleção do conteúdo educativo.
 * @returns Elemento de diálogo modal ou null quando fechada.
 */
export function HowItWorksModal({
  isOpen,
  onClose,
  screen,
}: HowItWorksModalProps) {
  const { t } = useTranslation("howItWorks");
  const [activeTab, setActiveTab] = useState<
    "overview" | "technical" | "steps"
  >("overview");

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const rawSteps = t(`screens.${screen}.steps`, { returnObjects: true });
  const steps: string[] = Array.isArray(rawSteps) ? rawSteps : [];

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/50 p-4 backdrop-blur-sm transition-opacity"
      role="presentation"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <Card
        className="flex max-h-[90vh] w-full max-w-2xl flex-col overflow-hidden border border-border bg-surface p-0 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="how-it-works-title"
      >
        <header className="flex items-start justify-between border-b border-border p-6 pb-4">
          <div>
            <h2
              id="how-it-works-title"
              className="text-xl font-bold tracking-tight text-foreground"
            >
              {t(`screens.${screen}.title`)}
            </h2>
            <p className="mt-1 text-sm text-muted">
              {t(`screens.${screen}.subtitle`)}
            </p>
          </div>
          <button
            type="button"
            className="rounded-lg p-1.5 text-muted transition-colors hover:bg-border/60 hover:text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            onClick={onClose}
            aria-label={t("close")}
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </header>

        <div className="flex border-b border-border bg-background/50 px-6">
          <button
            type="button"
            className={`mr-6 border-b-2 py-3 text-sm font-semibold transition-colors ${
              activeTab === "overview"
                ? "border-primary text-primary"
                : "border-transparent text-muted hover:text-foreground"
            }`}
            onClick={() => setActiveTab("overview")}
          >
            {t("tabs.overview")}
          </button>
          <button
            type="button"
            className={`mr-6 border-b-2 py-3 text-sm font-semibold transition-colors ${
              activeTab === "technical"
                ? "border-primary text-primary"
                : "border-transparent text-muted hover:text-foreground"
            }`}
            onClick={() => setActiveTab("technical")}
          >
            {t("tabs.technical")}
          </button>
          <button
            type="button"
            className={`border-b-2 py-3 text-sm font-semibold transition-colors ${
              activeTab === "steps"
                ? "border-primary text-primary"
                : "border-transparent text-muted hover:text-foreground"
            }`}
            onClick={() => setActiveTab("steps")}
          >
            {t("tabs.steps")}
          </button>
        </div>

        <div className="overflow-y-auto p-6 text-sm leading-relaxed text-foreground">
          {activeTab === "overview" ? (
            <div className="space-y-4">
              <p className="text-base text-foreground/90">
                {t(`screens.${screen}.overview`)}
              </p>
            </div>
          ) : null}

          {activeTab === "technical" ? (
            <div className="space-y-4">
              <p className="text-foreground/90">
                {t(`screens.${screen}.technical`)}
              </p>
            </div>
          ) : null}

          {activeTab === "steps" ? (
            <ol className="space-y-3">
              {steps.map((step, index) => (
                <li key={index} className="flex items-start gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                    {index + 1}
                  </span>
                  <span className="pt-0.5 text-foreground/90">{step}</span>
                </li>
              ))}
            </ol>
          ) : null}
        </div>

        <footer className="flex justify-end border-t border-border bg-background/50 p-4 px-6">
          <Button variant="default" size="sm" onClick={onClose}>
            {t("close")}
          </Button>
        </footer>
      </Card>
    </div>
  );
}
