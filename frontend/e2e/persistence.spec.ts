import { expect, test } from "@playwright/test";

test("@persistence keeps datasets, experiments, and analysis after restart", async ({
  page,
  request,
}) => {
  const datasets = await request.get("/api/v1/datasets");
  const experiments = await request.get("/api/v1/experiments");
  const decisionRuns = await request.get("/api/v1/decision-runs");
  const semanticSets = await request.get("/api/v1/semantic-features");
  const cascades = await request.get("/api/v1/confidence-cascades");
  expect(datasets.ok()).toBe(true);
  expect(experiments.ok()).toBe(true);
  expect(decisionRuns.ok()).toBe(true);
  expect(semanticSets.ok()).toBe(true);
  expect(cascades.ok()).toBe(true);
  expect((await datasets.json()) as unknown[]).not.toHaveLength(0);
  const history = (await experiments.json()) as {
    id: string;
    status: string;
    runs: { id: string; status: string }[];
  }[];
  expect(history).not.toHaveLength(0);

  const completedExperiment = history.find((experiment) =>
    experiment.runs.some((run) => run.status === "completed"),
  );
  const completedRun = completedExperiment?.runs.find(
    (run) => run.status === "completed",
  );
  expect(completedExperiment).toBeTruthy();
  expect(completedRun).toBeTruthy();
  expect((await decisionRuns.json()) as { status: string }[]).toContainEqual(
    expect.objectContaining({ status: "completed" }),
  );
  expect(
    (await semanticSets.json()) as { status: string; processed_rows: number }[],
  ).toContainEqual(
    expect.objectContaining({ status: "completed", processed_rows: 30 }),
  );
  expect((await cascades.json()) as unknown[]).not.toHaveLength(0);
  const analysis = await request.get(
    `/api/v1/experiments/runs/${completedRun!.id}/analysis`,
  );
  expect(analysis.ok()).toBe(true);

  await page.addInitScript(() => localStorage.setItem("uiLanguage", "en-US"));
  await page.goto(`/experiments/${completedExperiment!.id}`);
  await expect(
    page.getByText("Completed", { exact: true }).first(),
  ).toBeVisible();

  await page.goto("/decision-lab/benchmark");
  await expect(page.getByText("Is this customer active?")).toBeVisible();
  await page.goto("/decision-lab/semantic-features");
  await expect(page.getByText("Completed", { exact: true })).toBeVisible();
  await page.getByText("View questions, file, and details").click();
  await page.getByText("Processing metrics").click();
  await expect(page.getByText(/^Completed · 30\/30/)).toBeVisible();
  await page.goto("/decision-lab/cascades");
  await expect(page.getByText("Auto 0.90 · review 0.60")).toBeVisible();
});
