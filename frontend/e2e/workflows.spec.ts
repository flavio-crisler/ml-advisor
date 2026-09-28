import { expect, test, type Page } from "@playwright/test";
import path from "node:path";

const attendanceCsv = path.resolve("../backend/tests/fixtures/attendance.csv");
const irisCsv = path.resolve("../backend/tests/fixtures/kaggle_iris.csv");
const salaryCsv = path.resolve("../backend/tests/fixtures/kaggle_salary.csv");
const invalidCsv = path.resolve("../e2e/fixtures/invalid.csv");
const semanticCsv = path.resolve("../e2e/fixtures/support_semantic.csv");
const salesCsv = path.resolve(
  "../backend/tests/fixtures/synthetic_sales_semantic.csv",
);

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("uiLanguage", "en-US");
    localStorage.setItem("llmLanguage", "en-US");
  });
});

async function upload(page: Page, file: string) {
  await page.goto("/");
  await page.getByLabel("Select CSV file").setInputFiles(file);
  await expect(page).toHaveURL(/\/datasets\/[\w-]+$/);
}

async function waitForCompleted(page: Page) {
  await expect(
    page
      .getByRole("heading", { level: 1 })
      .locator("..")
      .getByText("Completed", { exact: true }),
  ).toBeVisible({ timeout: 90_000 });
}

test("Expert imports, trains, clones, edits, and compares a binary experiment", async ({
  page,
}) => {
  await upload(page, attendanceCsv);
  await expect(page.getByText("12").first()).toBeVisible();
  await expect(
    page.getByText(/customer_id looks like an identifier/i),
  ).toBeVisible();

  await page.getByRole("button", { name: "Configure manually" }).click();
  await expect(
    page.getByRole("heading", { name: /configure every experiment stage/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("checkbox", { name: "customer_id" }),
  ).not.toBeChecked();
  await expect(
    page.getByRole("checkbox", { name: "constant" }),
  ).not.toBeChecked();
  await page.getByLabel("Objective").fill("Predict missed appointments");
  await page.getByRole("button", { name: "Create advanced draft" }).click();

  await expect(page).toHaveURL(/\/experiments\/[\w-]+$/);
  await expect(page.getByText("Advanced", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Train" }).click();
  await waitForCompleted(page);

  await page
    .getByRole("button")
    .filter({ hasText: "logistic_regression" })
    .click();
  await expect(
    page.getByRole("navigation", { name: "Result sections" }),
  ).toContainText("Artifacts");
  await expect(page.locator("#metrics")).toBeVisible();
  await page.getByRole("button", { name: "Back to experiment" }).click();

  await page.getByRole("button", { name: "Clone as draft" }).click();
  const editor = page.getByLabel("Experiment specification JSON");
  const clonedSpec = JSON.parse(await editor.inputValue()) as {
    models: { id: string; parameters: Record<string, unknown> }[];
  };
  clonedSpec.models[0].parameters = { C: 0.5 };
  await editor.fill(JSON.stringify(clonedSpec, null, 2));
  await page.getByRole("button", { name: "Save configuration" }).click();
  await page.getByRole("button", { name: "Train" }).click();
  await waitForCompleted(page);
  await expect(
    page.getByRole("heading", { name: "Parent comparison" }),
  ).toBeVisible();
});

test("Expert trains selected free text with fold-fitted local TF-IDF", async ({
  page,
}) => {
  await upload(page, semanticCsv);
  await page.getByRole("button", { name: "Configure manually" }).click();
  await page.getByLabel("Outcome column").selectOption("target");
  await page
    .getByLabel("Objective")
    .fill("Predict cancellation from customer message");
  const message = page.getByRole("checkbox", { name: /message \(text/ });
  await expect(message).toBeDisabled();
  await page
    .getByRole("checkbox", { name: "Text vectorization (TF-IDF)" })
    .check();
  await expect(message).toBeEnabled();
  await message.check();
  await page.getByRole("button", { name: "Create advanced draft" }).click();
  await expect(page).toHaveURL(/\/experiments\/[\w-]+$/);
  await expect(
    page.getByText("Local TF-IDF (fitted in each training run)"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Train" }).click();
  await waitForCompleted(page);
});

test("Assisted creates a reviewed draft with the deterministic LLM", async ({
  page,
}) => {
  await upload(page, attendanceCsv);
  await page
    .getByLabel("What do you want to predict?")
    .fill("Predict missed appointments with a review step");
  await page.getByLabel("Advisor response language").selectOption("en-US");
  await page.getByRole("button", { name: "Generate recommendation" }).click();

  await expect(page.getByText(/transparent binary baseline/i)).toBeVisible();
  await expect(page.getByText(/no dataset rows were sent/i)).toBeVisible();
  await page.getByRole("button", { name: "Create guided draft" }).click();
  await expect(page.getByText("AI-guided", { exact: true })).toBeVisible();
  await expect(page.getByText("Draft", { exact: true })).toBeVisible();
});

test("Advisor repairs an invalid proposal and trains support_semantic", async ({
  page,
}) => {
  await upload(page, semanticCsv);
  await page
    .getByLabel("What do you want to predict?")
    .fill("Predict target using message and segment");
  await page.getByRole("button", { name: "Generate recommendation" }).click();

  await expect(
    page.getByText("Compare a text-aware baseline after validation."),
  ).toBeVisible();
  const proposal = JSON.parse(
    await page.getByLabel("Review and edit the proposal").inputValue(),
  ) as {
    target: string;
    features: string[];
    preprocessing: { id: string; strategy: string }[];
  };
  expect(proposal.target).toBe("target");
  expect(proposal.features).toEqual(["message", "segment"]);
  expect(proposal.preprocessing).toContainEqual(
    expect.objectContaining({ id: "text_vectorization", strategy: "tfidf" }),
  );
  expect(proposal.preprocessing.map((step) => step.id)).not.toContain("target");

  await page.getByRole("button", { name: "Create guided draft" }).click();
  await expect(page).toHaveURL(/\/experiments\/[\w-]+$/);
  await expect(page.getByText("Draft", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Train" }).click();
  await waitForCompleted(page);
  await expect(
    page.getByText("Local TF-IDF (fitted in each training run)"),
  ).toBeVisible();
  await expect(
    page.getByRole("button").filter({ hasText: "svm_classifier" }),
  ).toContainText("Completed");
  await page
    .getByRole("button")
    .filter({ hasText: "logistic_regression" })
    .click();
  await expect(page.locator("#metrics")).toBeVisible();
});

test("Auto trains a multiclass experiment with explicit averaging", async ({
  page,
}) => {
  await upload(page, irisCsv);
  await page
    .getByLabel("What do you want to predict?")
    .fill("Classify the iris species");
  await page.getByRole("button", { name: "Train automatically" }).click();

  await expect(page.getByText("Automatic", { exact: true })).toBeVisible();
  await expect(page.getByText("Multiclass classification")).toBeVisible();
  await waitForCompleted(page);
  await expect(
    page.getByRole("link", { name: "logistic_regression", exact: true }),
  ).toBeVisible();
});

test("Expert trains a regression experiment and excludes the identifier", async ({
  page,
}) => {
  await upload(page, salaryCsv);
  await page.getByRole("button", { name: "Configure manually" }).click();
  await page.getByLabel("Task").selectOption("regression");
  await page.getByLabel("Outcome column").selectOption("Salary");
  await page.getByLabel("Objective").fill("Estimate salary from experience");
  await expect(
    page.getByRole("checkbox", { name: "Unnamed: 0" }),
  ).not.toBeChecked();
  await page.getByRole("button", { name: "Create advanced draft" }).click();
  await expect(page.getByText("Regression", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Train" }).click();
  await waitForCompleted(page);
  await expect(
    page.getByRole("link", { name: "linear_regression", exact: true }),
  ).toBeVisible();
});

test("shows safe errors for invalid CSV and invalid LLM output", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Select CSV file").setInputFiles(invalidCsv);
  await expect(page.getByText("The CSV could not be analyzed.")).toBeVisible();

  await page.getByLabel("Select CSV file").setInputFiles(attendanceCsv);
  await page
    .getByLabel("What do you want to predict?")
    .fill("[invalid-llm] return a broken recommendation");
  await page.getByRole("button", { name: "Generate recommendation" }).click();
  await expect(
    page.getByText(/provider returned an invalid recommendation/i),
  ).toBeVisible();
});

test("switches the complete interface language independently", async ({
  page,
}) => {
  await upload(page, attendanceCsv);
  await page.getByRole("button", { name: "PT" }).click();
  await expect(
    page.getByRole("heading", {
      name: /agora escolha o que você deseja prever/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByLabel("Idioma da resposta do Advisor").first(),
  ).toHaveValue("en-US");
});

test("runs a zero-shot benchmark, opens calibration, and simulates a cascade", async ({
  page,
}) => {
  await upload(page, attendanceCsv);
  await page.getByRole("button", { name: "AI Decisions" }).click();
  await page.getByRole("link", { name: /Compare zero-shot/ }).click();
  await page.getByLabel("Dataset").selectOption({ label: "attendance" });
  await page.getByLabel("Outcome to predict (target)").selectOption("target");
  await page.getByRole("checkbox", { name: "active", exact: true }).check();
  await page
    .getByLabel("Question the model should answer")
    .fill("Is this customer active?");
  await page.getByLabel(/I understand that the selected columns/).check();
  await page.getByRole("button", { name: "Start comparison" }).click();
  await expect(page.getByText("Comparison added to the queue.")).toBeVisible();
  await expect(page.getByText("Completed", { exact: true })).toBeVisible({
    timeout: 90_000,
  });
  await page
    .getByRole("link")
    .filter({ hasText: "Is this customer active?" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Calibration Lab" }),
  ).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Brier Score")).toBeVisible();

  await page.getByRole("link", { name: /Set confidence/ }).click();
  await expect(
    page.getByRole("button", { name: "Simulate and save policy" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Simulate and save policy" }).click();
  await expect(page.getByLabel("Cascade simulation results")).toBeVisible();
  await page
    .getByRole("slider", { name: /Explore auto-accept threshold/ })
    .fill("0.5");
  await expect(
    page.getByText(
      "Threshold exploration uses persisted benchmark outputs and never calls the Decision Provider.",
    ),
  ).toBeVisible();
});

test("offers every compatible zero-shot target beyond profile suggestions", async ({
  page,
}) => {
  await upload(page, salesCsv);
  await page.getByRole("button", { name: "AI Decisions" }).click();
  await page.getByRole("link", { name: /Compare zero-shot/ }).click();
  await page
    .getByLabel("Dataset")
    .selectOption({ label: "synthetic_sales_semantic" });

  const returnFlag = page.getByRole("option", { name: "Return_Flag" });
  await expect(returnFlag).toBeEnabled();
  await expect(
    page.locator(
      'optgroup[label="Other compatible columns"] option[value="Return_Flag"]',
    ),
  ).toHaveCount(1);
  await page
    .getByLabel("Outcome to predict (target)")
    .selectOption("Return_Flag");
  await expect(page.getByLabel("Outcome to predict (target)")).toHaveValue(
    "Return_Flag",
  );
});

test("offers categorical text for semantic features without identifiers", async ({
  page,
}) => {
  await upload(page, salesCsv);
  await page.getByRole("button", { name: "AI Decisions" }).click();
  await page.getByRole("link", { name: /Create AI variables/ }).click();
  await page
    .getByLabel("Dataset")
    .selectOption({ label: "synthetic_sales_semantic" });

  await expect(
    page.getByRole("checkbox", { name: "Return_Reason" }),
  ).toBeVisible();
  await expect(
    page.getByRole("checkbox", { name: "Product_Name" }),
  ).toBeVisible();
  await expect(
    page.getByRole("checkbox", { name: "Transaction_ID" }),
  ).toHaveCount(0);
  await expect(page.getByRole("checkbox", { name: "Order_ID" })).toHaveCount(0);
  await expect(
    page.getByText(/Identifiers excluded from semantic generation/),
  ).toContainText("Transaction_ID");
  await page.getByRole("checkbox", { name: "Return_Reason" }).check();
  await page.getByLabel("Modeling objective").fill("Predict returns");
  await expect(
    page.getByRole("button", { name: "Suggest questions with AI" }),
  ).toBeEnabled();
});

test("renames and removes an experiment from the training history", async ({
  page,
}) => {
  await upload(page, attendanceCsv);
  await page.getByRole("button", { name: "Configure manually" }).click();
  await page.getByLabel("Objective").fill("Manage this history entry");
  await page.getByRole("button", { name: "Create advanced draft" }).click();

  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByLabel("Search").fill("Manage this history entry");
  await expect(page.getByText("Manage this history entry")).toBeVisible();

  await page.getByRole("button", { name: "Rename" }).click();
  await page.getByLabel("Experiment name").fill("Managed history draft");
  await page.getByRole("button", { name: "Save name" }).click();
  await expect(page.getByText("Managed history draft")).toBeVisible();

  await page.getByRole("button", { name: "Delete" }).click();
  await expect(
    page.getByRole("alertdialog", {
      name: "Remove experiment from history?",
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Remove from history" }).click();
  await expect(page.getByRole("alertdialog")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: /Managed history draft/ }),
  ).toHaveCount(0);
});

test("reviews and generates a frozen semantic feature set", async ({
  page,
}) => {
  await upload(page, semanticCsv);
  await page.getByRole("button", { name: "AI Decisions" }).click();
  await page.getByRole("link", { name: /Create AI variables/ }).click();
  await page.getByLabel("Dataset").selectOption({ label: "support_semantic" });
  await page.getByLabel("message").check();
  await page.getByLabel("Modeling objective").fill("Predict cancellation");
  await page.getByRole("button", { name: "Suggest questions with AI" }).click();
  await expect(page.getByLabel("Question cancellation_intent")).toHaveValue(
    "Does this customer express an intention to cancel?",
  );
  await page.getByLabel(/I understand that only the selected columns/).check();
  await page
    .getByRole("button", { name: "Generate reviewed variables" })
    .click();
  await expect(
    page.getByText("Semantic feature generation queued."),
  ).toBeVisible();
  await expect(page.getByText("Completed", { exact: true })).toBeVisible({
    timeout: 90_000,
  });
  await page.getByText("View questions, file, and details").click();
  await page.getByText("Processing metrics").click();
  await expect(page.getByText(/^Completed · 30\/30/)).toBeVisible();
  await expect(page.getByText(/jev-e2e-resolved/)).toBeVisible();
});
