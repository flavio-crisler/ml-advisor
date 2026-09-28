# Decision Providers

Decision Providers answer narrow typed questions with probability distributions. They are
separate from generative `LLMProvider` implementations: LLMs propose or explain, while Decision
Models produce semantic judgments that deterministic Python validates and measures.

The provider-neutral contract defines three question types:

- **Noul** returns a probability for a binary yes/no judgment.
- **Choice** sends a criteria map and returns a probability distribution across its alternatives.
- **Score** returns a probability distribution across ordered numeric levels and a deterministic
  expected score.

TypeSafe and OpenRouter adapters share the System One request shape. TypeSafe uses
`https://api.typesafe.ai/v1/systemone`; OpenRouter uses
`https://openrouter.ai/api/v1/systemone`. Both adapters persist the requested alias and the
resolved model returned by the provider. Choice and Score confidence is provider-derived and is
not assumed to equal the winning probability. Noul has no separate upstream confidence, so the
internal contract derives it deterministically from its distance to `0.5`.

The registry is available at `GET /api/v1/decision-providers`; an explicit connection check uses
`POST /api/v1/decision-providers/test`. The registry reports safe capabilities, configuration
state, external processing status, and the configured model. It never returns API keys. Decision
features default to disabled and do not affect existing experiment workflows.

Configure future adapters with the `DECISION_*` variables in `.env.example`. The requested model
and the provider-resolved model are separate fields so an alias is never the only reproducibility
record.

Only `DecisionState.values` is sent externally. Row identifiers and unselected dataset columns are
excluded. HTTP 429, 529, and transient transport failures use bounded exponential backoff. Current
contracts were reviewed on 2026-09-21 against the
[TypeSafe API reference](https://docs.typesafe.ai/api),
[TypeSafe model reference](https://docs.typesafe.ai/models), and
[OpenRouter System One guide](https://openrouter.ai/docs/guides/community/typesafe-sdk).

## Decision Lab workflows

`POST /api/v1/decision-benchmarks` queues a frozen Noul binary or Choice multiclass benchmark.
Dataset integrity, target exclusion, selected inputs, and the exact target class contract are
validated before provider calls. Metrics and calibration are computed locally; row responses use
Parquet, while spec, usage, summaries, reliability, and lifecycle metadata remain reopenable. The
Calibration Lab uses those artifacts for reliability, confidence, threshold, cost, and latency
plots without another external request.

Semantic Feature Discovery first asks the configured `LLMProvider` for validated question
proposals over explicitly selected free-text columns. After user review, the Decision Provider
generates reusable numeric artifacts. Noul produces one probability, Choice one probability per
choice, and Score an expected value plus uncertainty. Target-agnostic discovery is supported;
target-aware discovery fails closed until a development/test split is explicit.

Confidence Cascade stores an immutable policy and simulates auto-accept, review, or abstention from
a completed benchmark. Coverage, quality, review rate, cost when known, and threshold sensitivity
are deterministic calculations over persisted values. Slider changes never call Jev.

Live calls require private credentials and explicit external-processing acknowledgement. Mocks
exercise the full contract in automated tests; use `make test-decision` and `make test-semantic`.
