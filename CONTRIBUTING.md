# Contributing

## Start a development environment

Install Docker with Compose and GNU Make, then run:

```bash
cp .env.example .env
make setup
make dev
```

The Vite UI runs at `http://localhost:5173`; FastAPI runs at `http://localhost:8000`. Keep credentials only in the ignored `.env` file.

## Architecture boundaries

Keep `backend/app/domain/` independent of FastAPI, SQLAlchemy, pandas, and provider SDKs. Application services coordinate interfaces; implementations belong in `infrastructure/`, `ml/`, or `llm/`. Add ML components and LLM providers through adapters and registries. Never evaluate generated code or model output with `eval` or `exec`.

Flyway is the sole schema authority. Add forward-only migrations such as `backend/migrations/V4__add_run_notes.sql`; never call `create_all()` at runtime. Executed experiments are immutable: clone before changing a specification.

## Style and tests

Python uses four spaces, Ruff, `snake_case` functions/modules, and `PascalCase` classes. TypeScript/TSX uses two spaces, ESLint, Prettier, `PascalCase` components, and `useCamelCase` hooks.

Run the narrowest relevant suite while developing, then:

```bash
make test
make lint
make build
```

Run `make test-e2e` for workflow, persistence, API, or container changes. Every feature needs tests; every bug fix needs a regression test. Mock external LLM calls in automated tests. Update generated API types with `make generate-client` after changing OpenAPI contracts.

## Pull requests

Use short imperative commits such as `Add ordinal datetime preprocessing`. Keep commits focused. A pull request should state the problem and approach, identify changed contracts or migrations, link issues, list exact verification, and include screenshots for visible UI changes. Add both `pt-BR` and `en-US` strings when changing user-facing copy.
