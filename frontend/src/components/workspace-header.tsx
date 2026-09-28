import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLocation, useNavigate } from "react-router-dom";

import { Button } from "./ui/button";

type HealthState = "checking" | "online" | "offline";

/**
 * Cabeçalho global da aplicação com navegação entre módulos, verificação de saúde da API e seletor de idiomas.
 *
 * @returns Elemento header com controles de navegação e status do sistema.
 */
export function WorkspaceHeader() {
  const { i18n, t } = useTranslation("common");
  const location = useLocation();
  const navigate = useNavigate();
  const [health, setHealth] = useState<HealthState>("checking");
  const lastPath = readLastWorkspacePath();
  const isExperiments = location.pathname.startsWith("/experiments");
  const isDecisionLab = location.pathname.startsWith("/decision-lab");
  const isDatasets = !isExperiments && !isDecisionLab;

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/v1/health", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("health_check_failed");
        setHealth("online");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError")
          return;
        setHealth("offline");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (location.pathname === "/") return;
    if (!isWorkspacePath(location.pathname)) return;
    localStorage.setItem("mlAdvisorLastWorkspacePath", location.pathname);
  }, [location.pathname]);

  const lastArea = lastPath?.startsWith("/decision-lab")
    ? t("navigation.decisionLab")
    : lastPath?.startsWith("/experiments")
      ? t("navigation.experiments")
      : t("navigation.datasets");

  const changeLanguage = (locale: "pt-BR" | "en-US") => {
    void i18n.changeLanguage(locale);
    localStorage.setItem("uiLanguage", locale);
    document.documentElement.lang = locale;
  };

  return (
    <header className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-5">
      <div className="flex flex-wrap items-center gap-5">
        <button
          className="font-mono text-sm font-semibold text-primary"
          onClick={() => navigate("/")}
        >
          {t("brand.name")}
        </button>
        <nav
          className="flex items-center gap-1"
          aria-label={t("navigation.label")}
        >
          <Button
            aria-current={isDatasets ? "page" : undefined}
            variant={isDatasets ? "default" : "outline"}
            size="sm"
            onClick={() => navigate("/")}
          >
            {t("navigation.datasets")}
          </Button>
          <Button
            aria-current={isExperiments ? "page" : undefined}
            variant={isExperiments ? "default" : "outline"}
            size="sm"
            onClick={() => navigate("/experiments")}
          >
            {t("navigation.experiments")}
          </Button>
          <Button
            aria-current={isDecisionLab ? "page" : undefined}
            variant={isDecisionLab ? "default" : "outline"}
            size="sm"
            onClick={() => navigate("/decision-lab")}
          >
            {t("navigation.decisionLab")}
          </Button>
        </nav>
        {location.pathname === "/" && lastPath ? (
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(lastPath)}
          >
            {t("navigation.resume", { area: lastArea })}
          </Button>
        ) : null}
      </div>
      <div className="flex items-center gap-4">
        <span
          className="flex items-center gap-2 text-xs text-muted"
          data-testid="health-status"
        >
          <span
            aria-hidden="true"
            className={`h-2 w-2 rounded-full ${
              health === "online"
                ? "bg-success"
                : health === "offline"
                  ? "bg-error"
                  : "bg-muted"
            }`}
          />
          {t(`health.${health}`)}
        </span>
        <div className="flex gap-2" aria-label={t("language.selector")}>
          <Button
            variant={i18n.language === "pt-BR" ? "default" : "outline"}
            size="sm"
            onClick={() => changeLanguage("pt-BR")}
          >
            {t("language.portuguese")}
          </Button>
          <Button
            variant={i18n.language === "en-US" ? "default" : "outline"}
            size="sm"
            onClick={() => changeLanguage("en-US")}
          >
            {t("language.english")}
          </Button>
        </div>
      </div>
    </header>
  );
}

/**
 * Valida se uma rota pertence a um módulo de trabalho válido da plataforma.
 *
 * @param path - Caminho da URL atual.
 * @returns Verdadeiro se for uma rota de workspace válida.
 */
function isWorkspacePath(path: string): boolean {
  return ["/datasets/", "/experiments", "/decision-lab"].some((prefix) =>
    path.startsWith(prefix),
  );
}

/**
 * Recupera o último caminho de navegação de trabalho persistido no localStorage.
 *
 * @returns Caminho da última rota de workspace visitada, ou null.
 */
function readLastWorkspacePath(): string | null {
  const path = localStorage.getItem("mlAdvisorLastWorkspacePath");
  return path && isWorkspacePath(path) ? path : null;
}

/**
 * Layout padrão do workspace com contêiner responsivo e cabeçalho global.
 *
 * @param props.children - Conteúdo da página a ser renderizado no layout.
 * @returns Elemento main contendo o cabeçalho e os elementos filhos.
 */
export function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen bg-background px-5 py-6 text-foreground sm:px-8 lg:px-12">
      <div className="mx-auto flex max-w-7xl flex-col gap-8">
        <WorkspaceHeader />
        {children}
      </div>
    </main>
  );
}
