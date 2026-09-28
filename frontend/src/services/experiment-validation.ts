import i18n from "../i18n";

/**
 * Converte códigos e padrões de problemas de validação de experimentos em mensagens legíveis e localizadas.
 *
 * @param issues - Lista de identificadores ou mensagens de problemas do backend.
 * @returns Mensagem consolidada contendo os textos explicativos separados por ponto e vírgula.
 */
export function describeExperimentIssues(issues: string[]): string {
  return issues
    .map((issue) => {
      const text = issue.match(
        /Text features (.+) require text_vectorization:tfidf/,
      );
      if (text) {
        return i18n.t("experiments:validation.textRequiresTfidf", {
          columns: text[1],
        });
      }
      if (issue.includes("text_vectorization:max_features"))
        return i18n.t("experiments:validation.tfidfLimit");
      if (issue.includes("text_vectorization:total_features"))
        return i18n.t("experiments:validation.tfidfTotalLimit");
      if (issue === "binary_target_requires_two_classes")
        return i18n.t("experiments:validation.binaryTarget");
      if (issue === "multiclass_target_requires_three_classes")
        return i18n.t("experiments:validation.multiclassTarget");
      if (issue === "regression_target_must_be_numeric")
        return i18n.t("experiments:validation.numericTarget");
      return i18n.t("experiments:validation.other", { issue });
    })
    .join("; ");
}
