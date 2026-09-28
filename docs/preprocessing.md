# Preprocessing

`PipelineCompiler` turns the ordered `ExperimentSpec.preprocessing` choices into fitted row and feature transformers. It fits only on each training fold, applies the fitted transforms to validation data, and stores the resolved configuration beside every run artifact.

## Components and strategies

| ID | Strategies |
| --- | --- |
| `missing_values` | `drop_rows`, `drop_columns`, `mean`, `median`, `mode`, `constant` |
| `categorical_encoding` | `one_hot`, `ordinal`, `passthrough` |
| `boolean_encoding` | `binary`, `passthrough` |
| `numeric_scaling` | `standard`, `min_max`, `robust`, `passthrough` |
| `datetime` | `components`, `ordinal`, `drop` |
| `outliers` | `none`, `quantile_clip`, `iqr` |
| `imbalance` | `none`, `class_weight` |
| `feature_selection` | `none`, `variance_threshold` |
| `text_vectorization` | `tfidf` (explicit opt-in; `max_features` 1–2,048 per column) |

Query `GET /api/v1/preprocessors` instead of duplicating these values in clients.

Free text is not passed directly to tabular estimators. Select `text_vectorization: tfidf` to fit a separate vocabulary per selected text column inside each training fold. The default is 256 terms per column and the current dense tabular adapter is capped at 2,048 text features overall; larger corpora need a future sparse-data adapter. An inferred type is a heuristic, not a promise that a field is suitable or available at prediction time. Review identifiers, timestamps, and target leakage before training.

With the default missing-value strategy, absent text is replaced with an empty string before TF-IDF; this does not copy another row's text into the missing field. An explicit constant strategy still uses its configured constant.

If short text was inferred as categorical, an advanced experiment may set `text_vectorization.parameters.columns` explicitly, for example `{"columns": ["message"]}`. Only selected categorical or text features qualify; numeric/date/boolean columns cannot be relabelled this way. The profile remains unchanged.

## Resolution rules

Omitted steps use safe defaults: median imputation, one-hot categorical encoding, binary booleans, datetime components, and no scaling/outlier handling/class weighting/feature selection. Models that require scaling upgrade numeric passthrough to `standard`. Categorical passthrough requires a model with native categorical support. When an experiment requests `class_weight`, compatible classifiers receive the calculated weights; models without weighting support, such as KNN, resolve that step to `none` instead of failing the run.

Feature selection always starts from the explicit `features` list. Target, ID, ignored, constant, and suspected identifier columns are not reintroduced by the compiler. Persisted `preprocessing.json` records requested and resolved strategies, capability adjustments, original/output features, model capabilities, and transformer settings. Use those fields when comparing runs whose models require different pipelines.

When adding a component, update its transformer, compiler mapping, registry metadata, API/frontend handling, and focused tests. Any data-derived state must be learned inside `fit`, never from a full dataset before the validation split.
