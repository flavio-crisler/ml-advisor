import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { copyFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";

const screenshots = path.resolve("../docs/assets/screenshots");
const assets = path.resolve("../docs/assets");
const attendanceCsv = path.resolve("../backend/tests/fixtures/attendance.csv");
const semanticCsv = path.resolve("../e2e/fixtures/support_semantic.csv");

test.describe.configure({ mode: "serial" });
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("uiLanguage", "en-US");
    localStorage.setItem("llmLanguage", "en-US");
    localStorage.setItem("mlAdvisorGuideDismissed", "true");
  });
});

async function capture(page: Page, name: string) {
  await mkdir(screenshots, { recursive: true });
  await page.screenshot({
    path: path.join(screenshots, `${name}.png`),
    animations: "disabled",
  });
}

async function upload(page: Page, file: string) {
  await page.goto("/");
  await page.getByLabel("Select CSV file").setInputFiles(file);
  await expect(page).toHaveURL(/\/datasets\/[\w-]+$/);
}

async function uploadViaApi(request: APIRequestContext, file: string) {
  const response = await request.post("/api/v1/datasets", {
    multipart: {
      file: {
        name: path.basename(file),
        mimeType: "text/csv",
        buffer: await readFile(file),
      },
    },
  });
  expect([200, 201]).toContain(response.status());
  return (await response.json()) as {
    dataset: { id: string; hash: string };
    profile: { profile_version: string };
  };
}

async function createSemanticBaseline(request: APIRequestContext) {
  const uploaded = await uploadViaApi(request, semanticCsv);
  const response = await request.post("/api/v1/experiments", {
    data: {
      mode: "expert",
      spec: {
        spec_version: "1.0",
        dataset_id: uploaded.dataset.id,
        dataset_hash: uploaded.dataset.hash,
        profile_version: uploaded.profile.profile_version,
        name: "Synthetic support baseline",
        objective: "Predict cancellation from the segment",
        task: "binary_classification",
        target: "target",
        features: ["segment"],
        column_roles: {
          message: "ignored",
          segment: "feature",
          target: "target",
        },
        models: [{ id: "logistic_regression", enabled: true, parameters: {} }],
        preprocessing: [
          { id: "categorical_encoding", strategy: "one_hot", parameters: {} },
        ],
        validation: {
          strategy: "stratified_kfold",
          test_size: 0.2,
          folds: 3,
          shuffle: true,
          random_seed: 41,
        },
        metrics: { primary: "accuracy", secondary: [] },
        tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
      },
    },
  });
  expect(response.status()).toBe(201);
  const experiment = (await response.json()) as { id: string };
  expect(
    (await request.post(`/api/v1/experiments/${experiment.id}/run`)).status(),
  ).toBe(202);
  await expect
    .poll(
      async () => {
        const result = await request.get(
          `/api/v1/experiments/${experiment.id}`,
        );
        return ((await result.json()) as { status: string }).status;
      },
      { timeout: 90_000 },
    )
    .toBe("completed");
  return experiment.id;
}

test("capture real product screens with isolated synthetic data and mock providers", async ({
  page,
  request,
}) => {
  await upload(page, attendanceCsv);
  const columnProfile = page
    .getByText("Column profile", { exact: true })
    .first();
  await expect(columnProfile).toBeVisible();
  await columnProfile.scrollIntoViewIfNeeded();
  await capture(page, "dataset-profile");

  await page.getByRole("button", { name: "Configure manually" }).click();
  await expect(
    page.getByRole("heading", { name: "Configure every experiment stage" }),
  ).toBeVisible();
  await page
    .getByRole("checkbox", { name: "Random Forest Classifier" })
    .check();
  await capture(page, "experiment-builder");
  await page
    .getByLabel("Objective")
    .fill("Predict missed appointments from synthetic data");
  await page.getByRole("button", { name: "Create advanced draft" }).click();
  await page.getByRole("button", { name: "Train" }).click();
  await expect(page.getByText("Training job completed.")).toBeVisible({
    timeout: 90_000,
  });
  await expect(
    page.getByRole("button").filter({ hasText: "random_forest_classifier" }),
  ).toBeVisible();
  await page
    .getByRole("heading", { name: "Model runs" })
    .evaluate((element) => element.scrollIntoView({ block: "center" }));
  await capture(page, "model-comparison-results");

  await page
    .getByRole("button")
    .filter({ hasText: "logistic_regression" })
    .click();
  await expect(page.locator("#charts")).toBeVisible();
  const shap = page.getByText("SHAP summary", { exact: true }).first();
  await expect(shap).toBeVisible();
  await shap.evaluate((element) => element.scrollIntoView({ block: "start" }));
  await capture(page, "shap-explainability");

  await page.goto("/decision-lab/benchmark");
  await expect(
    page.getByRole("heading", { name: "Compare a zero-shot model" }),
  ).toBeVisible();
  await page.getByLabel("Dataset").selectOption({ label: "attendance" });
  await page.getByLabel("Outcome to predict (target)").selectOption("target");
  await page.getByRole("checkbox", { name: "active", exact: true }).check();
  await page
    .getByLabel("Question the model should answer")
    .fill("Is this customer active?");
  await capture(page, "decision-benchmark");
  await page.getByLabel(/I understand that the selected columns/).check();
  await page.getByRole("button", { name: "Start comparison" }).click();
  await expect(page.getByText("Completed", { exact: true })).toBeVisible({
    timeout: 90_000,
  });
  await page
    .getByRole("link")
    .filter({ hasText: "Is this customer active?" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Calibration Lab" }),
  ).toBeVisible();
  await expect(page.getByText("Brier Score")).toBeVisible();
  await capture(page, "calibration-lab");
  await page.getByRole("link", { name: /Set confidence/ }).click();
  await page.getByRole("button", { name: "Simulate and save policy" }).click();
  const cascadeResults = page.getByLabel("Cascade simulation results");
  await expect(cascadeResults).toBeVisible();
  await cascadeResults.scrollIntoViewIfNeeded();
  await capture(page, "confidence-cascade");

  await createSemanticBaseline(request);
  await page.goto("/decision-lab/semantic-features");
  await expect(
    page.getByRole("heading", { name: "Create variables with AI" }),
  ).toBeVisible();
  await page.getByLabel("Dataset").selectOption({ label: "support_semantic" });
  await page.getByLabel("message").check();
  await page.getByLabel("Modeling objective").fill("Predict cancellation");
  await page.getByRole("button", { name: "Suggest questions with AI" }).click();
  const suggestedQuestion = page.getByLabel("Question cancellation_intent");
  await expect(suggestedQuestion).toBeVisible();
  await suggestedQuestion.scrollIntoViewIfNeeded();
  await capture(page, "semantic-feature-discovery");
  await page
    .getByLabel("Impact comparison")
    .selectOption({ label: "Synthetic support baseline" });
  await page.getByLabel(/I understand that only the selected columns/).check();
  await page
    .getByRole("button", { name: "Generate reviewed variables" })
    .click();
  await expect(
    page.getByText(
      /Comparison with experiment Synthetic support baseline is complete/,
    ),
  ).toBeVisible({
    timeout: 120_000,
  });
  await expect(
    page.getByText("Impact of semantic variables on training"),
  ).toBeVisible({ timeout: 60_000 });
  await page
    .getByText("Impact of semantic variables on training")
    .locator("..")
    .screenshot({
      path: path.join(screenshots, "semantic-feature-impact.png"),
      animations: "disabled",
    });
});

test.describe("short product demo", () => {
  test("record dataset to model comparison", async ({ page }) => {
    await upload(page, attendanceCsv);
    await expect(page.getByText("Column profile")).toBeVisible();
    await page.waitForTimeout(750);
    await page.getByRole("button", { name: "Configure manually" }).click();
    await expect(
      page.getByRole("heading", { name: "Configure every experiment stage" }),
    ).toBeVisible();
    await page
      .getByRole("checkbox", { name: "Random Forest Classifier" })
      .check();
    await page.waitForTimeout(750);
    await page
      .getByLabel("Objective")
      .fill("Compare models on a synthetic attendance dataset");
    await page.getByRole("button", { name: "Create advanced draft" }).click();
    await page.waitForTimeout(750);
    await page.getByRole("button", { name: "Train" }).click();
    await expect(page.getByText("Training job completed.")).toBeVisible({
      timeout: 90_000,
    });
    await page.waitForTimeout(900);
    await page
      .getByRole("button")
      .filter({ hasText: "logistic_regression" })
      .click();
    await expect(page.locator("#metrics")).toBeVisible();
    await page.waitForTimeout(900);

    const video = page.video();
    expect(video).not.toBeNull();
    await page.close();
    await mkdir(assets, { recursive: true });
    await copyFile(
      await video!.path(),
      path.join(assets, "ml-advisor-demo.webm"),
    );
  });

  test("record baseline to semantic impact comparison", async ({
    page,
    request,
  }) => {
    const experimentId = await createSemanticBaseline(request);
    await page.goto(`/experiments/${experimentId}`);
    await expect(
      page.getByText("Completed", { exact: true }).first(),
    ).toBeVisible();
    await page.waitForTimeout(800);
    await page.getByRole("button", { name: "AI Decisions" }).click();
    await page.getByRole("link", { name: /Create AI variables/ }).click();
    await page
      .getByLabel("Dataset")
      .selectOption({ label: "support_semantic" });
    await page.getByLabel("message").check();
    await page.getByLabel("Modeling objective").fill("Predict cancellation");
    await page
      .getByRole("button", { name: "Suggest questions with AI" })
      .click();
    await expect(page.getByLabel("Question cancellation_intent")).toBeVisible();
    await page.waitForTimeout(800);
    await page.getByLabel("Impact comparison").selectOption(experimentId);
    await page
      .getByLabel(/I understand that only the selected columns/)
      .check();
    await page
      .getByRole("button", { name: "Generate reviewed variables" })
      .click();
    await expect(
      page.getByText(
        /Comparison with experiment Synthetic support baseline is complete/,
      ),
    ).toBeVisible({
      timeout: 120_000,
    });
    await page
      .getByText("Impact of semantic variables on training")
      .scrollIntoViewIfNeeded();
    await page.waitForTimeout(900);

    const video = page.video();
    expect(video).not.toBeNull();
    await page.close();
    await mkdir(assets, { recursive: true });
    await copyFile(
      await video!.path(),
      path.join(assets, "ml-advisor-semantic-demo.webm"),
    );
  });
});

test("render a GitHub social preview from versioned HTML/CSS", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 640 });
  await page.setContent(
    await readFile(path.resolve("../docs/social-preview.html"), "utf8"),
  );
  await expect(
    page.getByRole("heading", { name: /Compare models, inspect results/ }),
  ).toBeVisible();
  await mkdir(assets, { recursive: true });
  await page.screenshot({ path: path.join(assets, "social-preview.png") });
});
