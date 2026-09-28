import { expect, test, type APIRequestContext } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";

const attendanceCsv = path.resolve("../backend/tests/fixtures/attendance.csv");
const irisCsv = path.resolve("../backend/tests/fixtures/kaggle_iris.csv");
const smallCsv = path.resolve("../e2e/fixtures/small.csv");

type UploadedDataset = {
  dataset: { id: string; hash: string };
  profile: { profile_version: string };
};

async function upload(
  request: APIRequestContext,
  filePath: string,
): Promise<UploadedDataset> {
  const response = await request.post("/api/v1/datasets", {
    multipart: {
      file: {
        name: path.basename(filePath),
        mimeType: "text/csv",
        buffer: await readFile(filePath),
      },
    },
  });
  expect([200, 201]).toContain(response.status());
  return (await response.json()) as UploadedDataset;
}

function binarySpec(
  uploaded: UploadedDataset,
  overrides: Record<string, unknown> = {},
) {
  return {
    spec_version: "1.0",
    dataset_id: uploaded.dataset.id,
    dataset_hash: uploaded.dataset.hash,
    profile_version: uploaded.profile.profile_version,
    name: "E2E hardening experiment",
    objective: "Exercise a failure boundary safely",
    task: "binary_classification",
    target: "target",
    features: ["value"],
    column_roles: { value: "feature", target: "target" },
    models: [{ id: "logistic_regression", enabled: true, parameters: {} }],
    preprocessing: [],
    validation: {
      strategy: "train_test_split",
      test_size: 0.5,
      folds: 2,
      shuffle: true,
      random_seed: 41,
    },
    metrics: { primary: "accuracy", secondary: [] },
    tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
    ...overrides,
  };
}

async function createExperiment(request: APIRequestContext, spec: unknown) {
  const response = await request.post("/api/v1/experiments", {
    data: { mode: "expert", spec },
  });
  expect(response.status()).toBe(201);
  return (await response.json()) as { id: string };
}

test("rejects an unknown target before persisting the experiment", async ({
  request,
}) => {
  const uploaded = await upload(request, attendanceCsv);
  const spec = binarySpec(uploaded, {
    target: "missing_target",
    features: ["amount"],
    column_roles: { amount: "feature", missing_target: "target" },
  });
  const response = await request.post("/api/v1/experiments", {
    data: { mode: "expert", spec },
  });

  expect(response.status()).toBe(400);
  const payload = (await response.json()) as {
    detail: { issues: string[] };
  };
  expect(payload.detail.issues).toContain("unknown_target:missing_target");
});

test("accepts a CSV larger than the Nginx default request limit", async ({
  request,
}) => {
  const rowCount = 180_000;
  const csv = Buffer.from(`value,target\n${"1,yes\n".repeat(rowCount)}`);
  expect(csv.byteLength).toBeGreaterThan(1024 * 1024);

  const response = await request.post("/api/v1/datasets", {
    multipart: {
      file: {
        name: "proxy-large.csv",
        mimeType: "text/csv",
        buffer: csv,
      },
    },
  });

  expect(response.status()).toBe(201);
  const payload = (await response.json()) as UploadedDataset & {
    profile: { rows: number; size_bytes: number; profile_version: string };
  };
  expect(payload.profile.rows).toBe(rowCount);
  expect(payload.profile.size_bytes).toBe(csv.byteLength);
});

test("marks a too-small dataset experiment as failed without crashing the API", async ({
  request,
}) => {
  const uploaded = await upload(request, smallCsv);
  const experiment = await createExperiment(request, binarySpec(uploaded));
  const response = await request.post(
    `/api/v1/experiments/${experiment.id}/run`,
  );
  expect(response.status()).toBe(202);

  await expect
    .poll(
      async () => {
        const detail = await request.get(
          `/api/v1/experiments/${experiment.id}`,
        );
        return ((await detail.json()) as { status: string }).status;
      },
      { timeout: 30_000 },
    )
    .toBe("failed");
  expect((await request.get("/api/v1/health")).ok()).toBe(true);
});

test("records a model parameter failure and remains available", async ({
  request,
}) => {
  const uploaded = await upload(request, attendanceCsv);
  const spec = binarySpec(uploaded, {
    features: ["amount"],
    column_roles: {
      customer_id: "id",
      active: "ignored",
      created_at: "ignored",
      segment: "ignored",
      target: "target",
      amount: "feature",
      constant: "ignored",
    },
    models: [
      {
        id: "logistic_regression",
        enabled: true,
        parameters: { C: -1 },
      },
    ],
  });
  const experiment = await createExperiment(request, spec);
  expect(
    (await request.post(`/api/v1/experiments/${experiment.id}/run`)).status(),
  ).toBe(202);

  await expect
    .poll(
      async () => {
        const detail = await request.get(
          `/api/v1/experiments/${experiment.id}`,
        );
        return ((await detail.json()) as { status: string }).status;
      },
      { timeout: 30_000 },
    )
    .toBe("failed");
  expect((await request.get("/api/v1/health")).ok()).toBe(true);
});

test("cancels a queued or running training process", async ({ request }) => {
  const uploaded = await upload(request, irisCsv);
  const spec = {
    spec_version: "1.0",
    dataset_id: uploaded.dataset.id,
    dataset_hash: uploaded.dataset.hash,
    profile_version: uploaded.profile.profile_version,
    name: "Cancellable E2E experiment",
    objective: "Verify process cancellation",
    task: "multiclass_classification",
    target: "Species",
    features: [
      "SepalLengthCm",
      "SepalWidthCm",
      "PetalLengthCm",
      "PetalWidthCm",
    ],
    column_roles: {
      Id: "id",
      SepalLengthCm: "feature",
      SepalWidthCm: "feature",
      PetalLengthCm: "feature",
      PetalWidthCm: "feature",
      Species: "target",
    },
    models: [
      {
        id: "random_forest_classifier",
        enabled: true,
        parameters: { n_estimators: 5000 },
      },
    ],
    preprocessing: [],
    validation: {
      strategy: "stratified_kfold",
      test_size: 0.2,
      folds: 5,
      shuffle: true,
      random_seed: 41,
    },
    metrics: { primary: "accuracy", secondary: [] },
    tuning: { enabled: false, engine: "optuna", trials: 50, metric: null },
  };
  const experiment = await createExperiment(request, spec);
  expect(
    (await request.post(`/api/v1/experiments/${experiment.id}/run`)).status(),
  ).toBe(202);
  const cancellation = await request.post(
    `/api/v1/experiments/${experiment.id}/cancel`,
  );
  expect(cancellation.status()).toBe(200);
  expect((await cancellation.json()) as { status: string }).toMatchObject({
    status: "cancelled",
  });

  await expect
    .poll(async () => {
      const detail = await request.get(`/api/v1/experiments/${experiment.id}`);
      return ((await detail.json()) as { status: string }).status;
    })
    .toBe("cancelled");
});
