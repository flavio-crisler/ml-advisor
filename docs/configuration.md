# Configuration

Run `make setup`, edit the ignored `.env`, then start with `make up`. The production backend receives these variables through Compose's `env_file`. Changes require recreating the service (`docker compose up --build -d backend frontend`); restarting a container does not replace its environment. Never copy real credentials into `.env.example`.

The backend's source of truth is `backend/app/core/config/settings.py`; Compose also controls ports and container paths. The template lists intended local defaults. Paths refer to container filesystems, not host directories.

## Runtime and storage

| Variable | Template default | Effect |
| --- | --- | --- |
| `APP_ENV` | `development` | Environment label; does not enable authentication |
| `APP_NAME`, `APP_VERSION` | `ML Advisor`, `0.1.0` | Application/API metadata |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |
| `BACKEND_PORT`, `FRONTEND_PORT`, `VITE_PORT` | `8000`, `8080`, `5173` | Host ports bound to loopback; Vite only in development |
| `CORS_ORIGINS_RAW` | `http://localhost:5173,http://localhost:8080` | Comma-separated allowed browser origins; adjust for a changed UI port |
| `DATABASE_URL` | `sqlite:////data/database/app.db` | Compose explicitly fixes this value for the backend to match Flyway; changing only `.env` does not relocate the database |
| `DATASET_PATH`, `ARTIFACT_PATH` | `/data/datasets`, `/data/experiments` | Container storage directories under the mounted host `data/` |
| `MAX_UPLOAD_SIZE_MB` | `50` | Backend CSV upload limit |
| `DATASET_PREVIEW_ROWS` | `20` | Rows retained in the dataset profile preview; bounds available prompt samples |
| `DEFAULT_RANDOM_SEED` | `41` | Default seed used when assembling generative requests; experiment split/model settings are also persisted in the spec |
| `AUTH_MODE`, `ARTIFACT_STORAGE` | `disabled`, `local` | Configuration placeholders; the implementation has no auth adapter and always uses local filesystem storage |
| `UI_LANGUAGE`, `LLM_LANGUAGE` | `pt-BR`, `pt-BR` | Backend settings exist, but the current SPA does not consume them from Compose. UI/Advisor language is selected separately and stored in the browser |

The SPA's initial locale is `localStorage.uiLanguage`, then the build-time `VITE_UI_LANGUAGE`, then `pt-BR`. A `VITE_*` setting requires a frontend build/development configuration; editing the root backend `.env` does not inject it into an already-built SPA. Locale controls support `pt-BR` and `en-US`.

For alternate ports, edit `BACKEND_PORT`, `FRONTEND_PORT`, and, when relevant, `VITE_PORT` plus `CORS_ORIGINS_RAW`. Nginx and Vite proxy API requests internally, so no frontend API URL change is needed. `make generate-client` accepts `OPENAPI_URL` and `BACKEND_PORT` Make overrides, for example `make generate-client BACKEND_PORT=18000`; set this explicitly when regenerating types from a non-default port.

## Generative providers

| Variable | Template default | Effect |
| --- | --- | --- |
| `LLM_PROVIDER` | `openrouter` | `openrouter`, `ollama`, or `vllm` |
| `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` | Empty | Model, private credential, and optional endpoint override; a usable model/provider must be configured |
| `REASONING_MODEL` | `false` | OpenRouter: use deterministic `temperature=0`; `true` sends reasoning/verbosity instead |
| `LLM_REASONING`, `LLM_VERBOSITY` | `medium`, `low` | Reasoning-model settings; see [LLM providers](llm-providers.md) for payload behavior |
| `OPENROUTER_PROVIDER` | Empty | Optional ordered provider allowlist, without fallback outside that list |
| `LLM_SEND_SAMPLE` | `false` | Include selected preview rows in Advisor/semantic-question prompts when explicitly enabled |
| `LLM_SAMPLE_SIZE` | `10` | Sample cap, 1–100; actual rows cannot exceed the stored preview |
| `LLM_TIMEOUT_SECONDS` | `60` | Request timeout in seconds, greater than 0 and at most 600 |
| `ADVISOR_PROMPTS_PATH` | `/app/prompts` | Localized system-prompt directory in the image |

Advanced experiments, local preprocessing, training, and deterministic comparison work without a configured generative model. Recommendation, automatic configuration, optional explanations, improvement proposals, and semantic-question proposals require one. Provider configuration is described in [llm-providers.md](llm-providers.md).

## Decision Providers

| Variable | Template default | Effect |
| --- | --- | --- |
| `DECISION_FEATURES_ENABLED` | `false` | Enable optional Decision Lab capabilities |
| `DECISION_PROVIDER` | `typesafe_jev` | Configured adapter; current built-ins include `typesafe_jev` and `openrouter_jev` |
| `DECISION_MODEL` | `jev-latest` | Requested model alias; the resolved model is recorded separately |
| `DECISION_BASE_URL`, `DECISION_API_KEY` | Empty | Optional endpoint override and private credential |
| `DECISION_REQUEST_TIMEOUT_SECONDS` | `60` | Request timeout, greater than 0 and at most 600 seconds |
| `DECISION_MAX_RETRIES` | `3` | Bounded transient retries, 0–10 |
| `DECISION_CONCURRENCY` | `8` | Concurrent requests, 1–128; higher values can raise provider rate/cost pressure |
| `DECISION_CACHE_ENABLED` | `true` | Persist and reuse content-addressed row responses |
| `DECISION_EXTERNAL_PROCESSING_ACK_REQUIRED` | `true` | Require acknowledgement before configured external row processing |

Enabling the feature flag alone does not supply a working provider. Use the explicit connection check in the UI and review the selected columns before a benchmark or semantic generation. Calibration/cascade simulations read persisted responses and require no new row inference. See [Decision Providers](decision-providers.md).

## Data sent to providers

| Workflow | Provider payload |
| --- | --- |
| Advisor recommendation / automatic strategy | Objective, profile without preview, registry metadata; preview rows only with `LLM_SEND_SAMPLE=true` |
| Semantic-question proposal | Objective and profiles of selected columns; selected-column preview values only when sampling is enabled |
| Decision benchmark / semantic generation | Reviewed questions and selected cell values, excluding the target, unselected columns, and internal row identifiers; external acknowledgement applies |
| Model/chart explanation | Browser-assembled run metadata and chart aggregates; standard UI requests do not add raw dataset rows |
| Experiment-comparison / semantic-impact explanation | Server-assembled protocol and aggregate results; raw rows and local artifact paths are excluded |
| Supervised improvement proposal | Source experiment spec, source metrics, aggregate result context, and supported capabilities |
| Calibration / confidence-cascade sliders | Local computation over persisted probabilities |

Profiles may contain observed category values, labels, minima/maxima, and column names even with sampling disabled. Treat free-form objectives, column names, spec metadata, and aggregates as potentially sensitive too. The generic explanation/improvement API accepts caller-supplied context; the normal UI's aggregate-only behavior is not a universal server-side redaction guarantee for arbitrary API clients. These endpoints are part of the trusted local single-user interface.

`LLM_SEND_SAMPLE=false` does not disable Decision Provider row processing. Provider terms, data retention, and inference charges are external to this repository; review the configured provider before opting in. Automated suites use isolated mock services and never load the developer's provider credentials.
