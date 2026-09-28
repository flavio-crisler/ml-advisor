import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

import { App } from "./App";

const ExperimentsPage = lazy(() =>
  import("./features/experiments/experiments-page").then((module) => ({
    default: module.ExperimentsPage,
  })),
);
const ExperimentDetailPage = lazy(() =>
  import("./features/experiments/experiment-detail-page").then((module) => ({
    default: module.ExperimentDetailPage,
  })),
);
const RunDetailPage = lazy(() =>
  import("./features/experiments/run-detail-page").then((module) => ({
    default: module.RunDetailPage,
  })),
);
const ExpertExperimentPage = lazy(() =>
  import("./features/experiments/expert-experiment-page").then((module) => ({
    default: module.ExpertExperimentPage,
  })),
);
const DecisionLabPage = lazy(() =>
  import("./features/decision-providers/decision-lab-page").then((module) => ({
    default: module.DecisionLabPage,
  })),
);
const DecisionBenchmarkPage = lazy(() =>
  import("./features/decision-benchmark/decision-benchmark-page").then(
    (module) => ({ default: module.DecisionBenchmarkPage }),
  ),
);
const CalibrationLabPage = lazy(() =>
  import("./features/calibration-lab/calibration-lab-page").then((module) => ({
    default: module.CalibrationLabPage,
  })),
);
const SemanticFeaturePage = lazy(() =>
  import("./features/semantic-feature-discovery/semantic-feature-page").then(
    (module) => ({ default: module.SemanticFeaturePage }),
  ),
);
const ConfidenceCascadePage = lazy(() =>
  import("./features/confidence-cascade/confidence-cascade-page").then(
    (module) => ({ default: module.ConfidenceCascadePage }),
  ),
);

/**
 * Define a malha de roteamento principal da aplicação frontend.
 *
 * Mapeia as rotas de datasets, laboratório de decisão, calibração, descoberta semântica,
 * cascatas de confiança e gerenciamento de experimentos para seus respectivos componentes,
 * utilizando carregamento sob demanda (lazy loading) e Suspense.
 *
 * @returns Roteador configurado com todas as rotas da aplicação.
 */
export function AppRoutes() {
  return (
    <Suspense fallback={null}>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/datasets/:datasetId" element={<App />} />
        <Route
          path="/datasets/:datasetId/expert"
          element={<ExpertExperimentPage />}
        />
        <Route path="/experiments" element={<ExperimentsPage />} />
        <Route path="/decision-lab" element={<DecisionLabPage />} />
        <Route
          path="/decision-lab/benchmark"
          element={<DecisionBenchmarkPage />}
        />
        <Route
          path="/decision-lab/calibration/:runId"
          element={<CalibrationLabPage />}
        />
        <Route
          path="/decision-lab/semantic-features"
          element={<SemanticFeaturePage />}
        />
        <Route
          path="/decision-lab/cascades"
          element={<ConfidenceCascadePage />}
        />
        <Route
          path="/experiments/:experimentId"
          element={<ExperimentDetailPage />}
        />
        <Route
          path="/experiments/:experimentId/runs/:runId"
          element={<RunDetailPage />}
        />
      </Routes>
    </Suspense>
  );
}
