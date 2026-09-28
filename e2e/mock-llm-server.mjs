import { createServer } from "node:http";

const binaryRecommendation = {
  task: "binary_classification",
  target: "target",
  features: ["active", "created_at", "segment", "amount"],
  models: [{ id: "logistic_regression", enabled: true, parameters: {} }],
  preprocessing: [
    { id: "missing_values", strategy: "median", parameters: {} },
    { id: "categorical_encoding", strategy: "one_hot", parameters: {} },
    { id: "boolean_encoding", strategy: "binary", parameters: {} },
    { id: "numeric_scaling", strategy: "standard", parameters: {} },
    { id: "datetime", strategy: "components", parameters: {} },
  ],
  validation: {
    strategy: "train_test_split",
    test_size: 0.34,
    folds: 5,
    shuffle: true,
    random_seed: 41,
  },
  metrics: {
    primary: "f1",
    secondary: ["accuracy"],
    averaging: null,
    roc_auc_strategy: null,
    positive_label: "yes",
  },
  tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
  explanation: "Use a transparent binary baseline and review its measured F1 score.",
  assumptions: [],
  warnings: [],
};

const multiclassRecommendation = {
  task: "multiclass_classification",
  target: "Species",
  features: [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
  ],
  models: [{ id: "logistic_regression", enabled: true, parameters: {} }],
  preprocessing: [
    { id: "numeric_scaling", strategy: "standard", parameters: {} },
  ],
  validation: {
    strategy: "stratified_kfold",
    test_size: 0.2,
    folds: 3,
    shuffle: true,
    random_seed: 41,
  },
  metrics: {
    primary: "f1",
    secondary: ["accuracy"],
    averaging: "weighted",
    roc_auc_strategy: null,
    positive_label: null,
  },
  tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
  explanation: "Compare the three classes with explicit weighted averaging.",
  assumptions: [],
  warnings: [],
};

const regressionRecommendation = {
  task: "regression",
  target: "Salary",
  features: ["YearsExperience"],
  models: [{ id: "linear_regression", enabled: true, parameters: {} }],
  preprocessing: [
    { id: "numeric_scaling", strategy: "standard", parameters: {} },
  ],
  validation: {
    strategy: "kfold",
    test_size: 0.2,
    folds: 5,
    shuffle: true,
    random_seed: 41,
  },
  metrics: {
    primary: "rmse",
    secondary: ["mae", "r2"],
    averaging: null,
    roc_auc_strategy: null,
    positive_label: null,
  },
  tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
  explanation: "Measure the salary error with a linear regression baseline.",
  assumptions: [],
  warnings: [],
};

const supportRecommendation = {
  task: "binary_classification",
  target: "target",
  features: ["message", "segment"],
  models: [
    { id: "logistic_regression", enabled: true, parameters: {} },
    { id: "svm_classifier", enabled: true, parameters: {} },
  ],
  preprocessing: [
    { id: "text_vectorization", strategy: "tfidf", parameters: {} },
    { id: "categorical_encoding", strategy: "one_hot", parameters: {} },
    { id: "imbalance", strategy: "class_weight", parameters: {} },
  ],
  validation: {
    strategy: "stratified_kfold",
    test_size: 0.2,
    folds: 3,
    shuffle: true,
    random_seed: 41,
  },
  metrics: {
    primary: "accuracy",
    secondary: [],
    averaging: null,
    roc_auc_strategy: null,
    positive_label: null,
  },
  tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
  explanation: "Compare a text-aware baseline after validation.",
  assumptions: [],
  warnings: [],
};

function recommendationFor(payload) {
  const userMessage = payload.messages?.find((message) => message.role === "user");
  const advisorRequest = JSON.parse(userMessage?.content ?? "{}");
  if (Array.isArray(advisorRequest.selected_columns)) {
    const [column] = advisorRequest.selected_columns;
    return JSON.stringify({
      questions: [
        {
          id: "cancellation_intent",
          input_columns: [column],
          type: "noul",
          instructions: "Does this customer express an intention to cancel?",
          options: [],
        },
      ],
    });
  }
  if (advisorRequest.objective?.includes("[invalid-llm]")) {
    return "not valid JSON";
  }
  const columns = advisorRequest.dataset_profile?.column_names ?? [];
  if (columns.includes("Species")) return JSON.stringify(multiclassRecommendation);
  if (columns.includes("Salary")) return JSON.stringify(regressionRecommendation);
  if (columns.includes("message") && columns.includes("segment")) {
    if (!advisorRequest.validation_feedback) {
      return JSON.stringify({
        ...supportRecommendation,
        preprocessing: ["message", "segment", "target"].map((id) => ({
          id,
          strategy: "tfidf",
          parameters: {},
        })),
      });
    }
    return JSON.stringify(supportRecommendation);
  }
  return JSON.stringify(binaryRecommendation);
}

createServer((request, response) => {
  if (request.method === "GET" && request.url === "/health") {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end('{"status":"ok"}');
    return;
  }
  if (request.method === "GET" && request.url === "/v1/models") {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ models: [{ id: "jev-e2e-resolved" }] }));
    return;
  }
  if (request.method === "POST" && request.url === "/v1/systemone") {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => {
      try {
        const payload = JSON.parse(body);
        const answers = Object.fromEntries(
          Object.entries(payload.questions ?? {}).map(([id, question]) => [
            id,
            decisionAnswer(question, payload.state ?? {}),
          ]),
        );
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(
          JSON.stringify({
            id: "e2e-decision-request",
            model: "jev-e2e-resolved",
            answers,
            usage: { input_tokens: 12, output_tokens: 4, cost: 0.001 },
          }),
        );
      } catch {
        response.writeHead(400, { "Content-Type": "application/json" });
        response.end('{"error":"invalid decision test request"}');
      }
    });
    return;
  }
  if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
    response.writeHead(404);
    response.end();
    return;
  }
  let body = "";
  request.setEncoding("utf8");
  request.on("data", (chunk) => {
    body += chunk;
  });
  request.on("end", () => {
    try {
      const content = recommendationFor(JSON.parse(body));
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(
        JSON.stringify({
          choices: [{ message: { role: "assistant", content } }],
        }),
      );
    } catch {
      response.writeHead(400, { "Content-Type": "application/json" });
      response.end('{"error":"invalid test request"}');
    }
  });
}).listen(3000, "0.0.0.0");

function decisionAnswer(question, state) {
  if (question.type === "noul") {
    const text = JSON.stringify(state).toLowerCase();
    const positive =
      state.active === true || text.includes("cancel") || text.includes("terminate");
    return { type: "noul", noul: positive ? 0.9 : 0.1 };
  }
  if (question.type === "choice") {
    const choices = Object.keys(question.criteria ?? {});
    const probabilities = Object.fromEntries(
      choices.map((choice, index) => [
        choice,
        index === 0 ? 0.8 : 0.2 / Math.max(choices.length - 1, 1),
      ]),
    );
    return {
      type: "choice",
      choice: choices[0],
      probabilities,
      confidence: 0.7,
    };
  }
  if (question.type === "score") {
    return {
      type: "score",
      score: 0.8,
      probabilities: { 0: 0.2, 1: 0.8 },
      legend: { 0: "low", 1: "high" },
      confidence: 0.6,
    };
  }
  throw new Error("unsupported decision question");
}
