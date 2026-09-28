import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import enAdvisor from "./en-US/advisor.json";
import enCommon from "./en-US/common.json";
import enDatasets from "./en-US/datasets.json";
import enDecisionLab from "./en-US/decisionLab.json";
import enExperiments from "./en-US/experiments.json";
import enExpert from "./en-US/expert.json";
import enHowItWorks from "./en-US/howItWorks.json";
import enTraining from "./en-US/training.json";
import ptAdvisor from "./pt-BR/advisor.json";
import ptCommon from "./pt-BR/common.json";
import ptDatasets from "./pt-BR/datasets.json";
import ptDecisionLab from "./pt-BR/decisionLab.json";
import ptExperiments from "./pt-BR/experiments.json";
import ptExpert from "./pt-BR/expert.json";
import ptHowItWorks from "./pt-BR/howItWorks.json";
import ptTraining from "./pt-BR/training.json";

const defaultLanguage =
  localStorage.getItem("uiLanguage") ??
  import.meta.env.VITE_UI_LANGUAGE ??
  "pt-BR";

void i18n.use(initReactI18next).init({
  resources: {
    "en-US": {
      advisor: enAdvisor,
      common: enCommon,
      datasets: enDatasets,
      decisionLab: enDecisionLab,
      experiments: enExperiments,
      expert: enExpert,
      howItWorks: enHowItWorks,
      training: enTraining,
    },
    "pt-BR": {
      advisor: ptAdvisor,
      common: ptCommon,
      datasets: ptDatasets,
      decisionLab: ptDecisionLab,
      experiments: ptExperiments,
      expert: ptExpert,
      howItWorks: ptHowItWorks,
      training: ptTraining,
    },
  },
  lng: defaultLanguage,
  fallbackLng: "en-US",
  defaultNS: "common",
  interpolation: {
    escapeValue: false,
  },
});

document.documentElement.lang = defaultLanguage;

export default i18n;
