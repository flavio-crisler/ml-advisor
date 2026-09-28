# Stage 20: DecisionProvider Specification

## Intent

Add a provider-neutral contract for typed probabilistic decisions alongside, but independent
from, `LLMProvider`. The public question types are Noul, Choice, and Score. All contracts are
versioned, immutable Pydantic models with forbidden extra fields.

## Required behavior

- A `DecisionProvider` evaluates selected state fields against registered typed questions.
- Responses contain normalized probabilities, confidence, requested model, resolved model when
  available, usage, and latency metadata.
- The provider registry exposes capabilities and safe configuration status without credentials.
- With `DECISION_FEATURES_ENABLED=false`, all existing ML and LLM workflows remain operational.
- The frontend obtains provider metadata from `GET /api/v1/decision-providers`.

## Invariants

- Decision providers do not implement or inherit `LLMProvider`.
- Domain modules do not import FastAPI, SQLAlchemy, TypeSafe, or OpenRouter.
- Probability distributions sum to one. Choice and Score confidence remains provider-derived;
  Noul confidence is deterministically normalized as distance from 0.5.
- No API response, log field, representation, or persisted model contains an API key.
