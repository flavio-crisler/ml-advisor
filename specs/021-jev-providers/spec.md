# Stage 21: Jev Provider Specification

## Intent

Implement TypeSafe Jev and OpenRouter Jev as interchangeable `DecisionProvider` adapters over the
System One HTTP contract. Both adapters normalize Noul, Choice, and Score responses into the
provider-neutral domain models from Stage 20.

## Current external contract

Validated on 2026-09-21 against the TypeSafe API reference and OpenRouter TypeSafe SDK guide:

- TypeSafe base URL: `https://api.typesafe.ai`; endpoint: `POST /v1/systemone`.
- OpenRouter base URL: `https://openrouter.ai/api`; endpoint: `POST /v1/systemone`.
- Requests contain `model`, `state`, and `questions`.
- Responses contain `model`, `answers`, and `usage`; OpenRouter also exposes `id`, `provider`, and
  `usage.cost`.
- TypeSafe retries HTTP 429 and 529 with exponential backoff and honors `Retry-After` when present.
- The response `model` is the resolved model and must be persisted separately from the requested
  alias.

## Required behavior

- Serialize only `DecisionState.values`; never send `row_id` or unselected columns.
- Preserve TypeSafe Choice/Score confidence instead of replacing it with the largest probability.
- Derive Noul confidence deterministically as `abs(2 * probability_yes - 1)`.
- Normalize malformed responses, authentication failures, timeouts, rate limits, and overloads to
  safe application errors.
- Never include API keys or full row state in errors or logs.
- Support an explicit connection test and runtime provider selection through `DECISION_*` settings.
