#!/usr/bin/env sh
set -eu

project_name="ml-advisor-e2e"
compose_files="-f compose.yaml -f compose.e2e.yaml"

cleanup() {
  docker compose -p "$project_name" $compose_files down -v --remove-orphans
}

failure_logs() {
  docker compose -p "$project_name" $compose_files ps
  docker compose -p "$project_name" $compose_files logs --no-color --tail=200
}

trap 'failure_logs; cleanup' INT TERM HUP
test -f .env || cp .env.example .env
cleanup
docker compose -p "$project_name" $compose_files up --build -d

attempt=0
until curl --fail --silent http://localhost:28080/api/v1/health >/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    failure_logs
    cleanup
    exit 1
  fi
  sleep 1
done

if ! (
  cd frontend
  E2E_BASE_URL=http://localhost:28080 \
    PLAYWRIGHT_CHANNEL="${PLAYWRIGHT_CHANNEL-chrome}" \
    npm run test:e2e -- --grep-invert @persistence
); then
  failure_logs
  cleanup
  exit 1
fi

docker compose -p "$project_name" $compose_files restart backend frontend
attempt=0
until curl --fail --silent http://localhost:28080/api/v1/health >/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 60 ]; then
    failure_logs
    cleanup
    exit 1
  fi
  sleep 1
done

if ! (
  cd frontend
  E2E_BASE_URL=http://localhost:28080 \
    PLAYWRIGHT_CHANNEL="${PLAYWRIGHT_CHANNEL-chrome}" \
    npm run test:e2e -- --grep @persistence
); then
  failure_logs
  cleanup
  exit 1
fi

cleanup
