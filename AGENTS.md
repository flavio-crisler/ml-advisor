# Repository Guidelines

## Project Status & Structure

This repository implements a local, single-user pre-release ML workbench. Read `README.md` and `docs/index.md` for current capabilities and limitations. Public architecture decisions live in `docs/architecture.md` and `docs/adr/`. The original implementation plan lives under the mostly ignored `docs-implementacao/`; update it when an architectural decision changes, and also document the decision publicly.

FastAPI code lives in `backend/app/`, organized into `api/`, `application/`, `domain/`, `ml/`, `llm/`, and `infrastructure/`. Backend tests belong in `backend/tests/`, Flyway scripts in `backend/migrations/`, and localized prompts in `backend/prompts/`. React/Vite code lives in `frontend/src/features/`; component tests are colocated and browser tests live in `frontend/e2e/`. Container definitions belong in `docker/` and Compose files at the root.

## Build, Test, and Development Commands

The root `Makefile` is the supported Docker-first task interface. Run `make help` for all targets:

- `make setup` — create local configuration and data directories.
- `make up` / `make down` — start or stop the Compose stack.
- `make dev` — run the hot-reload development environment.
- `make test` — execute backend and frontend suites.
- `make lint` / `make format` — check or apply repository formatting.
- `make migrate` — apply Flyway migrations.
- `make generate-client` — regenerate TypeScript API types from OpenAPI.
- `make build` — build production images.
- `make test-e2e` / `make smoke` — run isolated browser suites with mock providers.

Documentation-only changes need no build step; verify relative links, distinguish historical checks from current evidence, and run `git diff --check`.

## Coding Style & Naming

Use four spaces in Python and two spaces in TypeScript/TSX. Prefer `snake_case` for Python modules and functions, `PascalCase` for classes and React components, and `useCamelCase` for hooks. Name migrations `V<number>__<snake_case_description>.sql`. Tooling uses pip, npm with `package-lock.json`, Ruff, ESLint, and Prettier; document changes before introducing alternatives.

Keep domain code independent of FastAPI and SQLAlchemy. Access ML implementations and LLM providers through adapters and registries. Never use `eval` or `exec` with model output. Put all user-facing strings in both `pt-BR` and `en-US` locale files.

## Testing Guidelines

Every feature needs tests; every bug fix needs a regression test. Test adapters in isolation and mock external LLM providers. Cover backend units, repositories, migrations, registries, and pipelines; cover frontend components, queries, forms, and i18n. Use Playwright for critical import-to-results workflows. Run the narrowest relevant suite during development and `make test` before review.

## Commits & Pull Requests

Use short, imperative subjects with the existing `feat:`, `fix:`, `docs:`, or `chore:` prefixes, and keep commits logically focused. Pull requests should explain the problem and approach, identify affected architecture or contracts, link relevant issues, list verification performed, and include screenshots for visible UI changes. Never stage `.env`, runtime data, provider credentials, or private artifacts; `.gitignore` does not remove historical blobs.
