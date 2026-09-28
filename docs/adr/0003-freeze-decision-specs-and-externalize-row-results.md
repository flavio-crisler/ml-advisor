# ADR 0003: Freeze Decision Specs and Externalize Row Results

Status: Accepted

## Context

Zero-shot evaluation must be reproducible and must not encourage question tuning against final
test outcomes. Per-row probabilities are also too large and access-heavy for the SQLite history
database, while summaries and lifecycle fields benefit from relational queries.

## Decision

Freeze the reviewed question, selected columns, provider, model, and evaluation contract when a
DecisionRun is submitted. Persist lifecycle, hashes, usage, and summary references in SQLite.
Persist row-level responses in Parquet and versioned specs/calibration/usage in JSON through
`ArtifactStorage`. Threshold and cascade exploration always reuses these persisted probabilities.

## Consequences

Changing a question creates a new run. SQLite remains compact, history is queryable, and
calibration/cascade experiments do not incur provider calls. Artifact storage and database records
must be backed up together.
