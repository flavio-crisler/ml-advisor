# Kaggle test datasets

- `kaggle_iris.csv`: downloaded from [`uciml/iris`](https://www.kaggle.com/datasets/uciml/iris) through the Kaggle dataset API. The publisher's page declares CC0.
- `kaggle_salary.csv`: downloaded from [`abhishek14398/salary-dataset-simple-linear-regression`](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression) through the Kaggle dataset API. The publisher's page declares CC0.

These small fixtures exercise all model adapters with real tabular data. Keep the filenames stable so the adapter integration test remains reproducible.

On 2026-09-28, Kaggle's anonymous dataset metadata endpoints for [Iris](https://www.kaggle.com/api/v1/datasets/view/uciml/iris) and [Salary](https://www.kaggle.com/api/v1/datasets/view/abhishek14398/salary-dataset-simple-linear-regression) both returned `licenseName: CC0: Public Domain`. This verifies the publishers' declared metadata, not a guarantee about all upstream rights. These third-party fixtures retain their declared terms separately from this repository's Apache-2.0 source license.

`attendance.csv` is a small synthetic test fixture: 12 patterned rows with numeric identifiers, booleans, invented segments, dates, amounts, and a binary target. It contains no production customer data. The browser fixtures under `e2e/fixtures/` are also synthetic, including the intentionally invalid CSV used for error handling.

`synthetic_sales_semantic.csv` is generated for this repository, not downloaded from Kaggle. Its 30 invented rows cover identifier exclusion, categorical text selection, missing return reasons, and a binary return target. Eight binary columns keep that target outside the profiler's first ten suggestions, so the compatible-target browser test exercises its intended case. It replaces the previously tracked large sales CSV whose redistribution terms were not documented. The removed file remains in old Git commits; review repository history before making the repository public.
