# Documentation

ML Advisor is an implemented, local, single-user pre-release workbench. The guides below describe the current source tree. Planning documents are not evidence that a feature works; dated reports describe only the checkout and environment tested at that time.

## Start and use

| Guide | Contents |
| --- | --- |
| [README](../README.md) | Quick start, capabilities, example workflow, and previews |
| [Configuration](configuration.md) | Environment variables, defaults, effective scope, and data sent to providers |
| [Docker operations](docker.md) | Stack lifecycle, migrations, ports, persistent data, backups, and diagnostics |
| [ExperimentSpec](experiment-spec.md) | Validated experiment contract and artifact contents |
| [Examples](../examples/README.md) | Structural JSON example and fixture locations |
| [Preprocessing](preprocessing.md) | Transformations, defaults, compatibility, and fold-fitted TF-IDF |
| [Results and AI improvements](results-and-improvements.md) | Execution rounds, deterministic ranking, optional explanations, and reviewed drafts |
| [LLM providers](llm-providers.md) | OpenRouter, Ollama, and vLLM setup and failure behavior |
| [Decision Providers](decision-providers.md) | Jev benchmarks, calibration, semantic features, and confidence cascades |

The running API serves Swagger at `/api/docs`, OpenAPI JSON at `/api/openapi.json`, and FastAPI's default ReDoc at `/redoc` on the backend port. Nginx proxies `/api/`, so Swagger and OpenAPI also work through the frontend URL. Route implementations and the served OpenAPI are authoritative for exact request/response schemas. The [dated endpoint inventory](../docs-implementacao/validacao-ponta-a-ponta-acompanhamento.md) records previous HTTP/browser checks.

## Contribute and extend

| Guide | Contents |
| --- | --- |
| [Architecture](architecture.md) | Dependency boundaries, jobs, storage, and lifecycle |
| [Architecture decisions](adr/) | Provider separation, HTTP adapters, frozen specs, and semantic artifacts |
| [Model adapters](model-adapters.md) | Supported model IDs and adapter extension procedure |
| [Internationalization](i18n.md) | UI namespaces, prompt locales, and parity checks |
| [Contributing](../CONTRIBUTING.md) | Development tools, tests, and pull requests |
| [Agent instructions](../AGENTS.md) | Repository conventions for coding agents |
| [Visual assets](visual-assets.md) | Synthetic screenshots, recordings, and regeneration |
| [Fixture provenance](../backend/tests/fixtures/KAGGLE_SOURCES.md) | Synthetic data and third-party sources |
| [Security](../SECURITY.md) | Local security model and private vulnerability reporting |
| [Changelog](../CHANGELOG.md) | Notable changes before the first stable tag |
| [Public release readiness](public-release-readiness.md) | Current evidence, historical validation, and publication blockers |

`docs-implementacao/` contains local plans, including the original MVP plan; most files there are intentionally ignored. Only the two reviewed endpoint-validation reports are versioned. Public setup and architecture must therefore remain complete in `README.md`, `docs/`, and the contribution guide without relying on local plans.

## Current limitations

- Authentication, distributed workers, model serving, and public-network deployment are outside this build. Changing `AUTH_MODE` does not implement authentication.
- Training and provider workers are local processes. Completed artifacts persist; jobs and SSE event buffers do not provide automatic continuation across restarts. Interrupted decision benchmarks are marked failed during startup.
- LLM schemas and registries constrain configuration, but generated prose is not checked exhaustively against numerical metrics. Natural-language requests do not impose hard training-cost limits; review the proposed models and tuning budget, or use advanced mode for an explicit configuration.
- Comparisons and SHAP are descriptive. A rank, a small delta, or high accuracy on a tiny fixture does not establish statistical significance, causality, or performance on new data.
- JavaScript chart code produces a large optional bundle. Python dependencies use version ranges rather than a complete lockfile, so fresh dependency resolution can differ between builds; saved runtime versions help identify what a run used.
- No stable API/artifact compatibility policy has been declared. Back up `data/` before upgrading; Flyway migrations are forward-only.
