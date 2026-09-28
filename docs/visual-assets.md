# Reproducing visual assets

Run `make docs-visuals` from the repository root. It starts a disposable Compose project on `127.0.0.1:28080`, uses the existing local LLM/Decision Provider mocks, records real UI interactions with a fixed 1440×900 browser viewport, and stops the isolated stack afterward. Install Chrome, or install Playwright Chromium with `cd frontend && npx playwright install chromium` and run `PLAYWRIGHT_CHANNEL='' make docs-visuals`.

The source data are `backend/tests/fixtures/attendance.csv` and `e2e/fixtures/support_semantic.csv`. Both are synthetic; no personal data or real provider credentials are used. The Compose test override does not load the developer's `.env`. The screenshots are not hand-composed results: the application imports, trains, evaluates, and renders them.

Generated files:

- `docs/assets/screenshots/*.png`: dataset profile, experiment builder, model results, SHAP, decision benchmark, calibration, semantic discovery and impact, confidence cascade.
- `docs/assets/ml-advisor-demo.webm`: dataset-to-results browser recording.
- `docs/assets/ml-advisor-semantic-demo.webm`: baseline-to-semantic-impact browser recording.
- `docs/assets/social-preview.png`: 1280×640 capture of `docs/social-preview.html`; upload it manually in the GitHub repository settings when publication is approved.

If `ffmpeg` is installed, the same command also creates optimized GIF versions of both recordings. Do not capture a personal local workspace for public documentation. These synthetic results demonstrate navigation and metric interpretation, not expected performance on real datasets.

Run `make smoke` for the separate short browser health suite; it does not replace `make test-e2e`.
