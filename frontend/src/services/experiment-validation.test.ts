import { describe, expect, it } from "vitest";

import i18n from "../i18n";
import { describeExperimentIssues } from "./experiment-validation";

describe("experiment validation messages", () => {
  it("explains text preprocessing requirements in both supported languages", async () => {
    const issues = [
      "preprocessing_incompatible:logistic_regression:Text features ['message'] require text_vectorization:tfidf",
    ];
    await i18n.changeLanguage("pt-BR");
    expect(describeExperimentIssues(issues)).toContain("exigem TF-IDF");
    await i18n.changeLanguage("en-US");
    expect(describeExperimentIssues(issues)).toContain("require TF-IDF");
  });
});
