import { useTranslation } from "react-i18next";
import { NavLink, useLocation } from "react-router-dom";

const items = [
  ["/decision-lab", "providers"],
  ["/decision-lab/benchmark", "benchmark"],
  ["/decision-lab/semantic-features", "semanticFeatures"],
  ["/decision-lab/cascades", "cascades"],
] as const;

/**
 * Barra de navegação interna do Decision Lab com links para Provedores, Benchmark, Features Semânticas e Cascatas.
 *
 * @returns Elemento nav estilizado com abas de navegação contextual.
 */
export function DecisionLabNav() {
  const { t } = useTranslation("decisionLab");
  const { pathname } = useLocation();

  return (
    <nav className="flex flex-wrap gap-2" aria-label={t("navigation.label")}>
      {items.map(([to, key]) => {
        const isBenchmarkActive =
          key === "benchmark" &&
          (pathname.startsWith("/decision-lab/benchmark") ||
            pathname.startsWith("/decision-lab/calibration"));

        return (
          <NavLink
            key={to}
            to={to}
            end={to === "/decision-lab"}
            className={({ isActive }) => {
              const active = isActive || isBenchmarkActive;
              return `flex flex-col rounded-xl border px-3 py-2 text-sm font-medium transition-colors ${
                active
                  ? "border-primary bg-primary text-white"
                  : "border-border bg-surface text-muted hover:border-primary hover:text-primary"
              }`;
            }}
          >
            <span>{t(`navigation.${key}`)}</span>
            <span className="text-[11px] font-normal opacity-75">
              {t(`navigation.${key}Technical`)}
            </span>
          </NavLink>
        );
      })}
    </nav>
  );
}
