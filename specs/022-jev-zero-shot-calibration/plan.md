# Stage 22 Implementation Plan

1. Add immutable decision/calibration entities, repositories, and Flyway persistence.
2. Execute selected rows through the provider registry with bounded concurrency, cache, and
   checkpoints.
3. Compute binary/multiclass metrics and calibration deterministically in Python.
4. Persist high-volume responses outside SQLite and publish run, event, cancel, and calibration APIs.
5. Add the localized benchmark and Calibration Lab screens with methodological separation.
6. Verify target exclusion, cache accounting, artifacts, metrics, plots, and disabled-feature safety.
