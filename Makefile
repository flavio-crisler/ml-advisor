.DEFAULT_GOAL := help

COMPOSE := docker compose
DEV_COMPOSE := $(COMPOSE) -f compose.yaml -f compose.dev.yaml
BACKEND_TEST_IMAGE := ml-advisor-backend-test
FRONTEND_DEV_IMAGE := ml-advisor-frontend-dev
BACKEND_PORT ?= 8000
OPENAPI_URL ?= http://host.docker.internal:$(BACKEND_PORT)/api/openapi.json

.PHONY: help setup up down restart logs build migrate migration-info generate-client \
	test test-backend test-frontend test-decision test-semantic lint lint-backend lint-frontend format \
	test-e2e smoke docs-visuals format-backend format-frontend dev ollama vllm clean

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Prepare local configuration and persistent directories
	@command -v docker >/dev/null || { echo "Docker is required."; exit 1; }
	@$(COMPOSE) version >/dev/null
	@test -f .env || cp .env.example .env
	@mkdir -p data/database data/datasets data/experiments
	@echo "Setup complete. Review .env, then run 'make up'."

up: setup ## Build and start the production-like stack
	$(COMPOSE) up --build -d

down: ## Stop the stack without deleting persisted data
	$(COMPOSE) down --remove-orphans

restart: down up ## Restart the stack

logs: ## Follow stack logs
	$(COMPOSE) logs -f

build: ## Build production backend and frontend images
	$(COMPOSE) build backend frontend

migrate: setup ## Apply Flyway migrations
	$(COMPOSE) run --rm flyway

migration-info: setup ## Show Flyway migration state
	$(COMPOSE) run --rm flyway -url=jdbc:sqlite:/data/database/app.db -locations=filesystem:/flyway/sql info

generate-client: ## Regenerate TypeScript types from the running backend OpenAPI schema
	docker build -f docker/Dockerfile.frontend --target development -t $(FRONTEND_DEV_IMAGE) .
	docker run --rm --user $$(id -u):$$(id -g) --add-host=host.docker.internal:host-gateway \
		-e OPENAPI_URL=$(OPENAPI_URL) \
		-v "$(CURDIR)/frontend:/app" -v /app/node_modules $(FRONTEND_DEV_IMAGE) npm run generate-client

test: test-backend test-frontend ## Run all test suites

test-backend: ## Run backend tests in Docker
	docker build -f docker/Dockerfile.backend --target test -t $(BACKEND_TEST_IMAGE) .
	docker run --rm $(BACKEND_TEST_IMAGE)

test-frontend: ## Run frontend tests in Docker
	docker build -f docker/Dockerfile.frontend --target development -t $(FRONTEND_DEV_IMAGE) .
	docker run --rm $(FRONTEND_DEV_IMAGE) npm test

test-decision: ## Run Decision Provider backend and frontend tests
	docker build -f docker/Dockerfile.backend --target test -t $(BACKEND_TEST_IMAGE) .
	docker run --rm $(BACKEND_TEST_IMAGE) pytest -q \
		tests/test_decision_domain.py tests/test_decision_provider_api.py tests/test_jev_providers.py \
		tests/test_decision_benchmarks.py tests/test_confidence_cascade.py
	docker build -f docker/Dockerfile.frontend --target development -t $(FRONTEND_DEV_IMAGE) .
	docker run --rm $(FRONTEND_DEV_IMAGE) npm test -- \
		src/features/decision-providers/decision-lab-page.test.tsx \
		src/features/decision-benchmark/decision-benchmark-page.test.tsx \
		src/features/confidence-cascade/confidence-cascade-page.test.tsx src/i18n/i18n.test.ts

test-semantic: ## Run semantic-variable discovery backend and frontend tests
	docker build -f docker/Dockerfile.backend --target test -t $(BACKEND_TEST_IMAGE) .
	docker run --rm $(BACKEND_TEST_IMAGE) pytest -q tests/test_semantic_features.py
	docker build -f docker/Dockerfile.frontend --target development -t $(FRONTEND_DEV_IMAGE) .
	docker run --rm $(FRONTEND_DEV_IMAGE) npm test -- \
		src/features/semantic-feature-discovery/semantic-feature-page.test.tsx src/i18n/i18n.test.ts

test-e2e: ## Run browser workflows against a clean Docker stack, restart it, and verify persistence
	@test -x frontend/node_modules/.bin/playwright || npm ci --prefix frontend
	./e2e/run-docker-e2e.sh

smoke: ## Visit principal screens on an isolated deterministic browser stack
	sh ./e2e/run-visual-suite.sh smoke

docs-visuals: ## Generate documentation screenshots and a short demo recording
	sh ./e2e/run-visual-suite.sh docs

lint: lint-backend lint-frontend ## Run all linters

lint-backend: ## Run Ruff checks in Docker
	docker build -f docker/Dockerfile.backend --target test -t $(BACKEND_TEST_IMAGE) .
	docker run --rm $(BACKEND_TEST_IMAGE) ruff check app tests

lint-frontend: ## Run ESLint and Prettier checks in Docker
	docker build -f docker/Dockerfile.frontend --target development -t $(FRONTEND_DEV_IMAGE) .
	docker run --rm $(FRONTEND_DEV_IMAGE) sh -c 'npm run lint && npm run format:check'

format: format-backend format-frontend ## Apply backend and frontend formatting

format-backend: ## Format Python sources with Ruff
	docker build -f docker/Dockerfile.backend --target test -t $(BACKEND_TEST_IMAGE) .
	docker run --rm -v "$(CURDIR)/backend:/app" $(BACKEND_TEST_IMAGE) ruff format app tests

format-frontend: ## Format frontend sources with Prettier
	docker build -f docker/Dockerfile.frontend --target development -t $(FRONTEND_DEV_IMAGE) .
	docker run --rm -v "$(CURDIR)/frontend:/app" -v /app/node_modules $(FRONTEND_DEV_IMAGE) npm run format

dev: setup ## Start the hot-reload development stack
	$(DEV_COMPOSE) up --build

ollama: setup ## Start the stack with the optional Ollama provider
	$(COMPOSE) --profile ollama up --build -d

vllm: setup ## Start the stack with the optional vLLM provider
	$(COMPOSE) --profile vllm up --build -d

clean: ## Stop containers and remove generated images (keeps data)
	$(COMPOSE) down --remove-orphans --rmi local
