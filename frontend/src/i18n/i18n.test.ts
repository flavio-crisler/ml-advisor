import { describe, expect, it } from "vitest";

import enAdvisor from "./en-US/advisor.json";
import enCommon from "./en-US/common.json";
import enDatasets from "./en-US/datasets.json";
import enDecisionLab from "./en-US/decisionLab.json";
import enExperiments from "./en-US/experiments.json";
import enExpert from "./en-US/expert.json";
import enTraining from "./en-US/training.json";
import ptAdvisor from "./pt-BR/advisor.json";
import ptCommon from "./pt-BR/common.json";
import ptDatasets from "./pt-BR/datasets.json";
import ptDecisionLab from "./pt-BR/decisionLab.json";
import ptExperiments from "./pt-BR/experiments.json";
import ptExpert from "./pt-BR/expert.json";
import ptTraining from "./pt-BR/training.json";

const en = {
  advisor: enAdvisor,
  common: enCommon,
  datasets: enDatasets,
  decisionLab: enDecisionLab,
  experiments: enExperiments,
  expert: enExpert,
  training: enTraining,
};
const pt = {
  advisor: ptAdvisor,
  common: ptCommon,
  datasets: ptDatasets,
  decisionLab: ptDecisionLab,
  experiments: ptExperiments,
  expert: ptExpert,
  training: ptTraining,
};

function keys(value: unknown, prefix = ""): string[] {
  if (!value || typeof value !== "object" || Array.isArray(value))
    return [prefix];
  return Object.entries(value).flatMap(([key, child]) =>
    keys(child, prefix ? `${prefix}.${key}` : key),
  );
}

describe("locale catalogs", () => {
  it("keeps every pt-BR and en-US key in parity", () => {
    expect(keys(pt).sort()).toEqual(keys(en).sort());
  });

  it("contains no blank user-facing translations", () => {
    for (const catalog of [en, pt]) {
      const values = JSON.stringify(catalog);
      expect(values).not.toMatch(/:\s*"\s*"/);
    }
  });
});
