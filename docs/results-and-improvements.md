# Results and AI improvements

Training persists metrics and artifacts independently of optional AI explanations. Tables, validation settings, and saved run metrics are the reference for measured results. Generated text can be useful for interpretation, but requires review.

## Compare models within an experiment

Each training invocation assigns a shared `execution_id` to its model runs. The comparison view selects one execution round and orders completed models by the experiment's primary metric: higher first for `maximize`, lower first for `minimize`. Failed, unfinished, missing-metric, and non-finite runs are listed as exclusions. Exact or numerically close metric ties share a rank; duration, model ID, and run ID make display order deterministic. A tie's first/last row is not evidence of a meaningful performance difference.

`run_index` belongs to an individual model's history and does not identify a shared round. Legacy runs without `execution_id` are not silently combined. The selected-round default prefers the most recent round with completed models; a caller may specify an explicit round.

Use `GET /api/v1/experiments/{experiment_id}/comparison` for the structured result and `POST /api/v1/experiments/{experiment_id}/explanation` for optional comparative prose. The latter requires at least two eligible models and a configured LLM; the backend builds the protocol/metric context and excludes raw rows, predictions, artifacts, and local paths. Ranking is computed before the LLM request. Between-experiment comparisons use `GET /api/v1/experiments/{experiment_id}/compare/{other_experiment_id}` and should be interpreted in light of each experiment's dataset, features, split, seed, and tuning budget.

## Explain a result or chart

`GET /api/v1/explanations/status` reports safe provider availability. The normal UI sends an explicit `POST /api/v1/explanations` only after the user requests an explanation. Whole-result explanations receive the aggregates of the displayed charts; **Deepen with AI** sends one chart's aggregates and its run context. Missing, duplicate, or invented chart IDs in the response are rejected.

Other supported explanation scopes cover calibration, confidence cascades, semantic impact, and experiment comparison. Dedicated semantic-impact and experiment-comparison routes build their context on the server. UI language changes require a new explanation in the selected language; there is no automatic promise that previously generated prose becomes a validated translation.

Response validation enforces structure and chart identity. It does not verify every numerical statement, inference about causality, or claim of statistical significance in generated prose. An earlier [real-provider validation report](../docs-implementacao/validacao-ponta-a-ponta-relatorio-final.md) recorded contradictory ROC-AUC numbers in one explanation. Check generated statements against the displayed measurements.

## Review an improvement and create a draft

The improvement action calls `POST /api/v1/improvement-proposals/supervised`. The backend loads the original spec/metrics, supplies its supported capability catalog, validates the proposed changes, and returns a reviewable diff with a proposed spec. Unsupported suggestions stay advisory; generated code is not executed.

After review, `POST /api/v1/improvement-proposals/supervised/draft` validates the source-spec hash and provider identity, persists a child experiment with `parent_experiment_id`, and records proposal provenance atomically. This creates a draft. Training is a separate explicit action; the original experiment and its measurements remain intact. Changing provider or source spec requires a new proposal.

Natural-language objectives are not hard cost controls. The Advisor can return more models or tuning trials than a prose request intended, while still meeting the typed contract. Review the resolved spec and tuning budget before execution; use advanced mode when those constraints must be explicit from the outset.

## Semantic comparisons and calibration

Semantic features are generated once as reviewed artifacts. Impact evaluation holds the configured dataset, split, seed, models, parameters, validation, and tuning budget constant across baseline and augmented arms. Eligible text may add a fold-fitted local TF-IDF arm. The optional impact explanation receives aggregate comparison metadata and limitations, not original text rows.

Calibration and confidence cascades reuse persisted probabilities. Threshold changes do not call Jev. Provider confidence, a predicted probability, and observed calibration are distinct values. Small synthetic datasets demonstrate behavior; they do not establish calibration quality, generalization, causal benefit, or statistical significance.

Relevant regression suites include `backend/tests/test_experiment_comparison.py`, `test_improvement_proposals.py`, `test_explanations.py`, `test_semantic_impact_explanation.py`, and their frontend feature/component counterparts. `make test-e2e` exercises the broader import-to-results workflows against a fresh isolated stack.
