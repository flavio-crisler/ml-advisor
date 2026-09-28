# Stage 22: Zero-Shot Benchmark and Calibration Specification

## Intent

Evaluate a frozen Jev Noul or Choice question against a labeled CSV without training the Decision
Model. Binary and multiclass results are persisted separately from supervised experiments and are
never presented as a like-for-like leaderboard.

## Required behavior

- The reviewed `ZeroShotDecisionSpec` fixes dataset, target, selected inputs, question, provider,
  requested model, averaging, and final-test semantics before execution.
- The target and every unselected column remain local; only selected values enter `DecisionState`.
- Python computes predictive, probability, calibration, latency, request, token, and cost metrics.
- Row outputs are Parquet artifacts; summaries, usage, spec, and calibration are JSON artifacts.
- History, progress, cancellation, and cache reuse remain available across new run submissions.
- Calibration charts and sliders use persisted output and do not call the provider again.

## Invariants

- Dataset bytes must match the persisted SHA-256 hash and use the profiled encoding/delimiter.
- Binary tasks require Noul and an existing positive class; multiclass tasks require Choice with the
  complete target class set and explicit averaging.
- Cost is absent when the provider cannot estimate it. Requested and resolved models are distinct.
