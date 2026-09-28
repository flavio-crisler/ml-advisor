# Docker Operations

Docker Compose 2.24.4 or newer is the supported runtime and CI interface; the isolated browser stack uses the [`!override` merge tag](https://docs.docker.com/reference/compose-file/merge/#replace-value). Run `make setup` to create the ignored `.env` without overwriting an existing file. See [configuration.md](configuration.md) for supported variables and their actual scope.

## Stacks

```bash
make up       # Flyway, production FastAPI image, Nginx SPA
make dev      # source mounts and backend/frontend hot reload
make down     # stop containers; keep ./data
make restart
make logs
```

Production-like defaults bind the UI on `127.0.0.1:FRONTEND_PORT` (8080) and API on `127.0.0.1:BACKEND_PORT` (8000). Development additionally binds Vite on `127.0.0.1:VITE_PORT` (5173). Optional Ollama and vLLM ports also bind to loopback. Nginx proxies `/api/`, so the browser does not need a separate API origin.

CSV uploads are limited by the backend setting `MAX_UPLOAD_SIZE_MB` (50 MB by default). Nginx does
not impose a second request-size limit and streams uploads to the backend. For a larger local
dataset, raise the value in `.env` according to the available disk and memory, then recreate the
backend with `docker compose up --build -d backend frontend`.

Flyway runs before backend health checks. Use `make migrate` to apply pending migrations and `make migration-info` to inspect state. Never edit an applied migration; add a new `V<number>__description.sql` file.

## Persistent data

The root `data/` bind mount contains:

- `database/app.db`: SQLite schema and history;
- `datasets/<dataset_id>/`: original CSV and profile;
- `experiments/<experiment_id>/`: immutable run artifacts.

`make down` and `make clean` keep this directory. Back up the whole directory together to preserve database-to-artifact references. The E2E stack uses a separate named volume and deletes it after the run.

For a consistent local backup, wait for active jobs to finish, then stop the stack and archive all of `data/` together:

```bash
make down
mkdir -p backups
tar -czf "backups/ml-advisor-data-$(date +%Y%m%d-%H%M%S).tar.gz" data
make up
```

`backups/` is ignored by Git and Docker. Keep `.env` in a separate private backup. To restore, stop the stack and restore the complete `data/` directory into an empty destination using a compatible checkout; retain the previous data until the restore is verified. Starting the stack applies pending forward migrations. Jobs in progress and in-memory SSE cursors do not survive a restart; completed results do.

## Configuration and diagnostics

Configuration comes from `.env`; Compose fixes container paths under `/data`. Set `LOG_LEVEL` to `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. Inspect health and logs with:

```bash
curl -fsS http://localhost:8000/api/v1/health
docker compose ps
docker compose logs --tail=200 backend
docker compose run --rm flyway -url=jdbc:sqlite:/data/database/app.db \
  -locations=filesystem:/flyway/sql info
```

Do not remove the loopback binding or expose this authentication-disabled MVP to untrusted networks. Keep API keys out of Compose files, images, logs, and version control. Optional Ollama and vLLM profiles are documented in [llm-providers.md](llm-providers.md).
