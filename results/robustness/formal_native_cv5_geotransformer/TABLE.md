# GeoTransformer robustness — native grouped CV5 × seed42

Each perturbation seed is averaged within biological fold first; values are the unweighted mean ± sample SD across the same five folds as the Main Benchmark. No shared-cohort rescoring is used.

| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| Clean | 0.00 | 16.52 ± 13.33% | 9.29 ± 6.98% | 100.00 ± 0.00% | 16.52 ± 13.33% |
| Coordinate noise | 0.02 | 16.82 ± 13.36% | 8.40 ± 6.81% | 100.00 ± 0.00% | 16.82 ± 13.36% |
| Coordinate noise | 0.05 | 12.48 ± 11.12% | 8.71 ± 6.53% | 100.00 ± 0.00% | 12.48 ± 11.12% |
| Coordinate noise | 0.10 | 8.00 ± 8.02% | 5.44 ± 2.44% | 100.00 ± 0.00% | 8.00 ± 8.02% |
| Coordinate noise | 0.20 | 5.28 ± 4.63% | 1.97 ± 1.82% | 100.00 ± 0.00% | 5.28 ± 4.63% |
| Missing neurons | 0.10 | 17.25 ± 13.31% | 10.06 ± 6.38% | 87.79 ± 2.72% | 15.31 ± 12.09% |
| Missing neurons | 0.20 | 16.83 ± 11.09% | 11.08 ± 7.73% | 80.36 ± 1.45% | 13.48 ± 8.94% |
| Missing neurons | 0.30 | 14.66 ± 10.56% | 9.99 ± 9.05% | 68.25 ± 2.53% | 10.10 ± 7.26% |
| Missing neurons | 0.40 | 15.34 ± 9.77% | 12.25 ± 7.86% | 57.91 ± 2.99% | 9.03 ± 6.03% |
| Missing neurons | 0.50 | 13.41 ± 13.02% | 11.15 ± 8.78% | 49.69 ± 2.22% | 6.78 ± 6.80% |
| Distractors | 0.10 | 16.35 ± 8.77% | 8.40 ± 5.78% | 100.00 ± 0.00% | 16.35 ± 8.77% |
| Distractors | 0.20 | 11.88 ± 7.87% | 6.36 ± 3.75% | 100.00 ± 0.00% | 11.88 ± 7.87% |
| Distractors | 0.30 | 10.87 ± 9.11% | 4.37 ± 2.69% | 100.00 ± 0.00% | 10.87 ± 9.11% |
| Distractors | 0.40 | 9.08 ± 6.58% | 2.85 ± 2.76% | 100.00 ± 0.00% | 9.08 ± 6.58% |
| Distractors | 0.50 | 8.60 ± 6.58% | 2.94 ± 2.39% | 100.00 ± 0.00% | 8.60 ± 6.58% |

Coverage is the surviving evaluable-query fraction relative to the clean fold. Activity noise is not applicable because GeoTransformer consumes geometry only.
