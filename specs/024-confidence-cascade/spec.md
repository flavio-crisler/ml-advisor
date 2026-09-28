# Stage 24: Confidence Cascade Specification

## Intent

Simulate an immutable operational policy over a completed DecisionRun. High-confidence rows are
accepted automatically; remaining rows go to manual review or abstention. No workforce queue or
fallback provider is included.

## Required behavior

- `CascadePolicy` validates ordered auto-accept/review thresholds and an enum action.
- Simulation loads persisted row confidence and correctness, never calls DecisionProvider, and
  computes coverage, auto-decision/review/abstain rates, auto accuracy/error, threshold
  sensitivity, confidence distribution, and cost when inputs are known.
- Policy and metrics are immutable JSON artifacts with durable history that can be reopened.
- The UI plots threshold sensitivity and confidence distribution. Moving its exploration slider
  selects a persisted sensitivity point without HTTP or provider calls.

## Invariants

- A cascade requires a completed DecisionRun with compatible response artifacts.
- Review cost is optional and is never invented.
- Saved simulation records cannot be updated or deleted through repository operations.
