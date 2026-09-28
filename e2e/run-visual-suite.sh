#!/usr/bin/env sh
set -eu

suite="${1:-}"
case "$suite" in
  smoke|docs) ;;
  *) echo "Usage: $0 smoke|docs" >&2; exit 2 ;;
esac

project_name="ml-advisor-$suite"
test -x frontend/node_modules/.bin/playwright || npm ci --prefix frontend
test -f .env || cp .env.example .env

cleanup() {
  docker compose -p "$project_name" -f compose.yaml -f compose.e2e.yaml down -v --remove-orphans
}

failure_logs() {
  docker compose -p "$project_name" -f compose.yaml -f compose.e2e.yaml ps
  docker compose -p "$project_name" -f compose.yaml -f compose.e2e.yaml logs --no-color --tail=120
}

trap 'failure_logs; cleanup' INT TERM HUP
cleanup
docker compose -p "$project_name" -f compose.yaml -f compose.e2e.yaml up --build -d

attempt=0
until curl --fail --silent http://localhost:28080/api/v1/health >/dev/null; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 90 ]; then
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
    npx playwright test -c "playwright.$suite.config.ts"
); then
  failure_logs
  cleanup
  exit 1
fi

if [ "$suite" = docs ] && command -v ffmpeg >/dev/null 2>&1; then
  for demo in ml-advisor-demo ml-advisor-semantic-demo; do
    ffmpeg -y -i "docs/assets/$demo.webm" \
      -vf "fps=8,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer" \
      -loop 0 "docs/assets/$demo.gif"
  done
fi

cleanup
