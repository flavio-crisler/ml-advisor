# ADR 0004: Precompute Semantic Features as Reusable Artifacts

Status: Accepted

## Context

Calling an external provider inside an ML transformer would hide network, cost, retry, and leakage
behavior inside training folds. It would also make repeated model comparisons non-deterministic.

## Decision

Review and freeze typed semantic questions before generation. Execute them once in a separate,
checkpointed application job, persist numeric features in row-aligned Parquet, and let the pure
`SemanticFeatureTransformer` only validate and attach those values. Persist question and feature
metadata beside the artifact. Target-aware proposal remains unavailable until an explicit
development/test split is modeled.

## Consequences

Many models can reuse one paid generation pass, retries can use the row cache, and baseline versus
augmented evaluation can keep every other condition fixed. Dataset row identity becomes part of
the artifact contract.
