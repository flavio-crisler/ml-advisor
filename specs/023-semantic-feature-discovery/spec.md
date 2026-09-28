# Stage 23: Semantic Feature Discovery Specification

## Intent

Turn explicitly selected free-text columns into reusable numeric features using reviewed, frozen
Noul, Choice, or Score questions. An existing `LLMProvider` proposes questions; the configured
`DecisionProvider` generates values.

## Required behavior

- Profiling identifies candidate free-text columns but never selects or sends them automatically.
- Target-agnostic proposal sends only selected schema/profile data and an optional selected-column
  sample. Dataset values are untrusted data and the LLM cannot generate executable code.
- Users can review, edit, add, or remove typed questions before generation.
- Generation groups compatible questions, sends only their declared input columns, checkpoints by
  chunk, reuses row cache, and persists aligned Parquet features plus metadata.
- Noul stores probability, Choice stores each class probability, and Score stores deterministic
  expected score and uncertainty.
- Optional impact evaluation holds dataset, split, seed, models, parameters, validation, and zero
  tuning budget constant; semantic columns are identified in SHAP output.

## Invariants

- Questions and provider/model are immutable once queued; retries create a new history entry.
- `SemanticFeatureTransformer` performs no network access and rejects missing/duplicate row IDs.
- Target-aware discovery is rejected until an explicit development/test split contract exists.
- Provider cost remains unavailable rather than inferred when usage metadata is absent.
