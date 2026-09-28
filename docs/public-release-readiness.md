# Public release readiness — 2026-09-28

## Current conclusion

The implemented local pre-release has substantial user, API, architecture, contribution, and security documentation. Setup, configuration scope, data sharing, model-comparison rounds, reviewed AI improvements, and limitations now have public guides in the [documentation index](index.md). **The existing Git history is not ready to make public while the historical sales CSV remains unresolved.** Technical checks and documentation do not settle that data-publication decision. No repository publication, history rewrite, push, tag, or GitHub release was performed by this audit.

## Current audit scope and findings

- Inspected tracked files, local changes, ignored paths, all locally available Git refs, Docker configuration/build inputs, fixture provenance, and the running API's 57 OpenAPI operations. The unauthenticated GitHub repository API returned 404; current visibility, private vulnerability reporting, branch protections, and remote-only refs were not confirmed. A 404 does not establish whether a private repository exists.
- Updated stale `AGENTS.md` planning/tooling claims, aligned the README clone URL with `origin`, added missing `CORS_ORIGINS_RAW` to the environment template, and documented actual configuration limits. In particular, changing `AUTH_MODE` does not add authentication, `ARTIFACT_STORAGE` does not select a remote adapter, and root backend locale settings do not configure the built SPA.
- Expanded `.gitignore` to protect all `data/` contents except the three scaffold `.gitkeep` files, SQLite3 sidecars and database backups, local exports/backups, nested credentials, notebook checkpoints, generic `node_modules/`, and editor/merge leftovers. Reviewed fixtures, generated API types, the npm lockfile, documentation, and public media remain publishable. No currently tracked path matches the ignore rules.
- Expanded `.dockerignore` for nested `.env` variants, credentials, virtual environments, browser output, local website/plans, public media, backups, and runtime data. An actual exported Docker build context excluded synthetic secret-file markers and local data while retaining required source, prompt, fixture, and lockfile inputs. No real secret values were printed or copied into public artifacts.
- Gitleaks v8.30.1, with default detection rules and full output redaction, found no credentials in the reviewed 388-file publishable snapshot or the **47 commits** reachable through locally available refs at the final scan (`c985bbb` was the checkout HEAD). Two local documentation commits appeared during the review and were included in the final scan. This checks known credential patterns, not dataset privacy or every possible secret. Unfetched remote refs and hosted copies were outside the scan.
- Confirmed historical blob `405dc20257322de46f71d391613edb1efb0f02dc`: `backend/tests/fixtures/Sales_transactions_2022_2025.csv`, **5,190,513 bytes**, still reachable through Git history. Its earlier removal from the current tree does not prevent download of the blob by someone receiving that history. The synthetic replacement is present in the current tree.
- Rechecked the publishers' fixture metadata: both small Kaggle datasets declare `CC0: Public Domain` through their anonymous metadata endpoints. See [fixture provenance](../backend/tests/fixtures/KAGGLE_SOURCES.md). This is a check of declared terms, not an independent rights guarantee for the historical sales file.
- Corrected ambiguous Playwright model-name assertions exposed by the new comparison table. The original regression assertion matched both a run link and a comparison label; the multiclass case had the same ambiguity. Both now assert the exact model run link, preserving the completed-training requirement.

## Verification in this audit

| Check | Current evidence |
| --- | --- |
| `make test` | 217 backend and 59 frontend tests passed |
| `make lint` | Ruff, ESLint, and Prettier passed |
| `make build` | Backend and frontend production images built, including a final build after Docker exclusion changes; optional chart bundle still produces a size warning |
| Fresh publishable-file snapshot | `make setup`, template `.env` creation, persistent-directory scaffolding, and Compose validation passed; not a clean-machine installation |
| Compose configurations | Production, development, and isolated E2E configurations validated |
| Git exclusions | 19 private/generated path cases and 14 publishable path cases passed; no ignored tracked files |
| Docker exclusions | 14 excluded paths and 7 required build inputs checked against an actual exported context |
| `make test-e2e` | 20 workflow/hardening/usability scenarios and 1 post-restart persistence test passed after fixing ambiguous selectors |
| `make smoke` | 1 isolated browser smoke test passed |
| Final Gitleaks scan | No detected credentials in 388 reviewed files or 47 reachable local-history commits |
| Relative Markdown links / `git diff --check` | 56 relative links resolved to publishable targets; whitespace check passed |

The test/lint/build checks above ran against the current source including the pre-existing frontend changes. ESLint and Prettier were also rerun on the corrected E2E file. Backend tests reported three upstream SHAP/Matplotlib deprecation warnings. Browser tests use mock providers and fresh disposable data. The [2026-09-26 report](../docs-implementacao/validacao-ponta-a-ponta-relatorio-final.md) records previous live-provider/HTTP/manual browser evidence; that evidence was not repeated with paid providers in this audit. The existing main stack and its user data were left running; test stacks removed only their own disposable volumes.

## Required before sharing publicly

1. **Resolve the historical sales file.** Establish permission/provenance, or publish a reviewed clean snapshot without this history, or deliberately remove the file from every ref intended for publication. Rewriting/pushing existing history needs coordination and was not authorized or executed here. `.gitignore` only prevents future additions.
2. **Verify the intended public object set.** Scan the final tree and all refs actually being published, including remote refs not present locally. The local scan cannot certify an unfetched branch or an already-hosted copy.
3. **Commit and review the intended snapshot.** Recheck `git status`, the staged file list, relative links, and setup against the exact commit being shared, including any final report edits. Keep local `.env`, databases, uploads, exports, and model artifacts out of the commit.
4. **Confirm the private reporting channel.** `SECURITY.md` asks for GitHub private vulnerability reporting when available, but no verified fallback contact is published. Enable the reporting feature, or add a real private contact, before public reports arrive. GitHub metadata and social-preview settings remain maintainer tasks.

For a fresh repository without the historical blobs, first review and commit the intended snapshot. Then export **only that reviewed commit's tracked tree** into a new sibling directory:

```bash
# Run only after the intended changes are committed and the tree is reviewed.
mkdir ../ml-advisor-public
git archive --format=tar HEAD | tar -xf - -C ../ml-advisor-public
```

The destination must be new; the export contains no `.git` or previous commits and excludes ignored, untracked local data. Validate it again before initializing a separate repository. Do not copy the whole workspace or `.git` into the export. This option preserves the existing private history; it is not a purge of any copy already hosted elsewhere.

## Stable-release work remains separate

There is no first stable tag or API/artifact compatibility guarantee. Repeat full gates at the selected release commit, run fresh-machine/manual smoke and relevant real-provider checks, review AI numerical/cost limitations, and settle versioning before tagging. Local single-user operation and loopback-bound ports remain the supported security model.

## Archived preparation report — 2026-09-24

The sections below are historical evidence and original pending items from 2026-09-24. They are not current gate results; use the audit above for present status.

**Status:** prepared for a future public repository, **not cleared for publication or a stable tag**. No tag, GitHub release, or publication was created. The existing Apache-2.0 license, application architecture, provider contracts, and `pt-BR`/`en-US` UI remain unchanged.

## Completed

- Audited the current tracked tree, prospective new text files, Git-tracked paths in history, ignore rules, and high-confidence credential patterns. No real credential, tracked `.env`, database, log, or local artifact was identified in the current tree. The only private-looking URL/key matches inspected were Docker-local or explicit test placeholders. This was a targeted audit, not a comprehensive secret-scanner certification.
- Replaced the undocumented 18,045-row sales CSV with `backend/tests/fixtures/synthetic_sales_semantic.csv` (30 invented rows). Updated the backend and browser tests that used it; no current source/test reference still requires the large file. Documented synthetic provenance and direct Kaggle source/license pages for the remaining small fixtures in `backend/tests/fixtures/KAGGLE_SOURCES.md`.
- Corrected the README clone URL, added a restrained pre-release notice, two selected previews, demo links, and reproduction commands. Added `SECURITY.md`, bug/feature issue forms, a PR template, Dependabot configuration, and an `Unreleased` changelog. No ceremonial code of conduct was added.
- Added an isolated Playwright documentation suite with mock providers, synthetic datasets, fixed browser viewport, and nine stable screenshots under `docs/assets/screenshots/`. It also generates `docs/assets/social-preview.png` (1280×640) from versioned HTML/CSS and two short WebM recordings. See `docs/visual-assets.md` for reproduction. `ffmpeg` was unavailable here, so no GIF was generated; the script will create optimized GIFs if it is installed.
- Added an independent `make smoke` browser suite. It checks main-screen rendering and navigation, JavaScript exceptions, and API 5xx responses without replacing the deeper E2E suite. Its disposable Compose stack does not load the developer's `.env`.

## Verification performed

| Command | Result |
| --- | --- |
| `make docs-visuals` | 4 Playwright tests passed; nine screenshots, two WebM demos, and social preview generated |
| `make test-e2e` | 18 workflow/hardening tests and 1 post-restart persistence test passed |
| `make test` | 189 backend and 41 frontend tests passed |
| `make lint` | Ruff, ESLint, and Prettier passed |
| `make build` | Backend and frontend production images built |
| `make smoke` | 1 browser smoke test passed |
| `git diff --check` | Passed |

The browser suites used local fake LLM/Decision Provider services. These results do **not** validate a real provider, a public deployment, or the planned manual release smoke tests.

## Risks and human decisions

1. **Historical CSV remains in Git history.** Deleting `Sales_transactions_2022_2025.csv` from the current tree does not remove its old blob. Its redistribution rights were not documented, and it contains customer-like fields. Before making this repository public, decide whether rights can be demonstrated or create a clean public history/remove that blob. Do not publish the existing history by default.
2. **Security review is bounded.** No dedicated history-wide secret scanner (for example, Gitleaks) was available. Run one on every branch and tag before publication, review any findings privately, and rotate any exposed credential rather than merely deleting it.
3. **Local-only security model.** Authentication is disabled. Keep published ports on loopback; do not deploy this build on an untrusted network without a separate security design. Enable GitHub private vulnerability reporting or provide a private contact channel before public issues arrive.
4. **Human-facing assets and claims.** Review screenshots/video for desired copy and visual quality, verify third-party fixture licensing/attribution once more, and upload the social preview manually in GitHub settings. The synthetic results illustrate workflow, not expected model quality. Review README wording before LinkedIn publication.
5. **Non-blocking presentation/performance follow-up.** The confidence-cascade threshold chart has crowded labels at the documentation viewport, and the frontend build reports a large JavaScript chunk. Neither failed the release-preparation checks, but both deserve review before using those views prominently in public material.

## Files removed or substituted

- Removed from the current tree: `backend/tests/fixtures/Sales_transactions_2022_2025.csv` (~5 MB; still present in old commits).
- Added: `backend/tests/fixtures/synthetic_sales_semantic.csv` (30 rows); updated its dependent tests and provenance documentation.

## Before making the repository public

- [ ] Resolve the historical sales CSV and run a history-wide credential/data scan on all refs; inspect the final public Git object set.
- [ ] Confirm fixture redistribution terms and attribution; remove any fixture whose rights cannot be established.
- [ ] Run manual smoke tests from a fresh clone, including real-provider opt-in/privacy checks; record failures and fixes.
- [ ] Verify README links and setup on a clean machine; review generated media and optionally install `ffmpeg` to produce compact GIFs.
- [ ] Configure private vulnerability reporting, repository metadata, issue labels, branch protections, and social preview on GitHub.
- [ ] Recheck that `.env`, uploaded datasets, SQLite files, model artifacts, logs, and local screenshots are not staged.

## Before the first stable tag (separate later gate)

- [ ] Finish manual functional smoke tests and address release-blocking defects.
- [ ] Run the complete test, lint, build, E2E, and browser smoke gates against the intended release commit.
- [ ] Validate at least one real configured provider and document any provider-specific limitations without committing credentials.
- [ ] Freeze the public API/artifact compatibility expectations, update `CHANGELOG.md` and versioned documentation, then decide the tag/version deliberately.
- [ ] Only after approval, create the tag and GitHub release with reviewed notes and assets.
