# Examples

`experiment-spec.binary.json` demonstrates the complete v1 contract. It is structurally valid, but its placeholder dataset ID, hash, and column names must be replaced with values from an uploaded dataset before experiment creation.

Validate it against the running API:

```bash
curl -sS -X POST http://localhost:8000/api/v1/experiment-specs/validate \
  -H 'Content-Type: application/json' \
  --data @examples/experiment-spec.binary.json
```

Small CSVs used for adapter integration tests are in `backend/tests/fixtures/`; their provenance is documented in `KAGGLE_SOURCES.md`.
