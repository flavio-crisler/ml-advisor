import { describe, expect, it } from "vitest";

import {
  buildPlotExplanationCharts,
  summarizePlotArtifacts,
  summarizeNumericDistribution,
  summarizeTuningTrials,
} from "./explanation-context";

describe("summarizePlotArtifacts", () => {
  it("keeps aggregate chart values and strips per-row vectors", () => {
    const result = summarizePlotArtifacts({
      confusion_matrix: {
        data: [
          {
            type: "heatmap",
            x: ["no", "yes"],
            y: ["no", "yes"],
            z: [
              [8, 1],
              [2, 9],
            ],
          },
        ],
      },
      actual_vs_predicted: {
        data: [
          {
            type: "scatter",
            x: [1000, 2000, 3000],
            y: [900, 2200, 3100],
          },
        ],
      },
      residual_plot: {
        data: [
          {
            type: "scatter",
            x: [0.8, 1.1, 1.4],
            y: [0.1, -0.2, 0.05],
          },
        ],
      },
      probability_distribution: {
        data: [
          {
            type: "histogram",
            name: "yes",
            x: [0.1, 0.2, 0.8, 0.9],
          },
        ],
      },
      shap_summary: {
        data: [
          {
            type: "scatter",
            name: "class-a",
            x: [0.2, -0.4],
            y: ["salary", "age"],
          },
        ],
      },
    });

    expect(result.confusion_matrix.traces[0]).toMatchObject({
      point_count: 2,
      x: ["no", "yes"],
      z: [
        [8, 1],
        [2, 9],
      ],
    });
    expect(result.actual_vs_predicted.traces[0]).toMatchObject({
      type: "scatter",
      name: null,
      point_count: 3,
      aggregation: "paired_numeric_statistics",
      paired: {
        count: 3,
        mean_y_minus_x: 66.66666666666667,
      },
    });
    expect(result.actual_vs_predicted.traces[0]).not.toHaveProperty("x");
    expect(result.residual_plot.traces[0]).toMatchObject({
      aggregation: "paired_numeric_statistics",
      paired: { count: 3 },
    });
    expect(result.residual_plot.traces[0].paired).not.toHaveProperty(
      "mean_y_minus_x",
    );
    expect(result.probability_distribution.traces[0]).toMatchObject({
      aggregation: "histogram_and_distribution_statistics",
      distribution: { count: 4 },
    });
    expect(result.probability_distribution.traces[0]).not.toHaveProperty("x");
    expect(result.shap_summary.traces[0]).toMatchObject({
      type: "scatter",
      name: "class-a",
      point_count: 2,
      aggregation: "per_feature_shap_statistics",
      feature_count: 2,
      omitted_feature_count: 0,
      features: [
        { feature: "age", mean_absolute_shap: 0.4 },
        { feature: "salary", mean_absolute_shap: 0.2 },
      ],
    });
    expect(result.shap_summary.traces[0]).not.toHaveProperty("x");
  });

  it("builds a stable chart contract and a reusable displayed histogram", () => {
    const charts = buildPlotExplanationCharts(
      {
        learning_curve: {
          data: [{ x: [10, 20], y: [0.7, 0.85], type: "scatter" }],
        },
      },
      (chartId) => ({
        title: `Title ${chartId}`,
        purpose: "Explain validation progress.",
        axes: { x: "Training rows", y: "Validation score" },
      }),
    );

    expect(charts).toEqual([
      {
        chart_id: "learning_curve",
        title: "Title learning_curve",
        purpose: "Explain validation progress.",
        axes: { x: "Training rows", y: "Validation score" },
        data_policy: "aggregate_only_no_raw_rows",
        data: {
          traces: [
            {
              type: "scatter",
              name: null,
              point_count: 2,
              aggregation: "displayed_aggregate_values",
              x: [10, 20],
              y: [0.7, 0.85],
            },
          ],
        },
      },
    ]);

    const distribution = summarizeNumericDistribution([0, 0, 10, 20]);
    expect(distribution.count).toBe(4);
    expect(distribution.bins.reduce((total, bin) => total + bin.count, 0)).toBe(
      4,
    );
  });

  it("caps high-dimensional SHAP context and reports omitted features", () => {
    const featureCount = 25;
    const result = summarizePlotArtifacts({
      shap_summary: {
        data: [
          {
            type: "scatter",
            x: Array.from({ length: featureCount }, (_, index) => index / 10),
            y: Array.from(
              { length: featureCount },
              (_, index) => `feature_${index}`,
            ),
          },
        ],
      },
    });
    const trace = result.shap_summary.traces[0] as {
      feature_count: number;
      features: Array<{ feature: string; mean_absolute_shap: number }>;
      omitted_feature_count: number;
    };

    expect(trace.feature_count).toBe(25);
    expect(trace.features).toHaveLength(20);
    expect(trace.omitted_feature_count).toBe(5);
    expect(trace.features[0]).toMatchObject({
      feature: "feature_24",
      mean_absolute_shap: 2.4,
    });
  });

  it("decodes binary encoded plotly buffer arrays (bdata)", () => {
    // [0.1, 0.2, 0.8, 0.9] as float64 base64
    const bdata = "mpmZmZmZuT+amZmZmZnJP5qZmZmZmek/zczMzMzM7D8=";
    const result = summarizePlotArtifacts({
      probability_distribution: {
        data: [
          {
            type: "histogram",
            name: "0",
            x: { dtype: "f8", bdata } as unknown as number[],
          },
        ],
      },
      roc_curve: {
        data: [
          {
            type: "scatter",
            name: "model",
            x: { dtype: "f8", bdata } as unknown as number[],
            y: { dtype: "f8", bdata } as unknown as number[],
          },
        ],
      },
    });

    expect(result.probability_distribution.traces[0]).toMatchObject({
      point_count: 4,
      aggregation: "histogram_and_distribution_statistics",
      distribution: {
        count: 4,
        statistics: {
          minimum: 0.1,
          maximum: 0.9,
        },
      },
    });
    expect(result.roc_curve.traces[0]).toMatchObject({
      point_count: 4,
      aggregation: "displayed_aggregate_values",
      x: [0.1, 0.2, 0.8, 0.9],
      y: [0.1, 0.2, 0.8, 0.9],
    });
  });

  it("summarizes tuning trials without sending every trial payload", () => {
    expect(
      summarizeTuningTrials([
        {
          status: "completed",
          metric_value: 0.8,
          parameters: { depth: 3, learning_rate: 0.1 },
        },
        {
          status: "pruned",
          metric_value: 0.6,
          parameters: { depth: 5, learning_rate: 0.05 },
        },
        { status: "failed", metric_value: null, parameters: { depth: 7 } },
      ]),
    ).toMatchObject({
      total_count: 3,
      status_counts: { completed: 1, pruned: 1, failed: 1 },
      metric_value_statistics: {
        minimum: 0.6,
        maximum: 0.8,
        mean: 0.7,
      },
      parameter_names: ["depth", "learning_rate"],
    });
  });
});
