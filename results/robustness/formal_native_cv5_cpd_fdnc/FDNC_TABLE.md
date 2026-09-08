# fDNC robustness — native grouped CV5 × 42

Within each fold, perturbation replicates are averaged first; values are the unweighted mean ± sample SD across five folds. No shared-cohort rescoring is used.

| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| Clean | 0.00 | 45.11 ± 16.99% | 51.49 ± 18.65% | 100.00 ± 0.00% | 45.11 ± 16.99% |
| Coordinate noise | 0.02 | 43.92 ± 15.85% | 50.88 ± 17.26% | 100.00 ± 0.00% | 43.92 ± 15.85% |
| Coordinate noise | 0.05 | 42.15 ± 13.82% | 46.49 ± 14.39% | 100.00 ± 0.00% | 42.15 ± 13.82% |
| Coordinate noise | 0.10 | 36.01 ± 13.36% | 39.48 ± 14.45% | 100.00 ± 0.00% | 36.01 ± 13.36% |
| Coordinate noise | 0.20 | 24.18 ± 8.94% | 26.48 ± 10.39% | 100.00 ± 0.00% | 24.18 ± 8.94% |
| Missing neurons | 0.10 | 44.34 ± 15.99% | 48.71 ± 16.31% | 89.70 ± 1.63% | 39.82 ± 14.55% |
| Missing neurons | 0.20 | 47.33 ± 13.75% | 50.80 ± 15.18% | 79.88 ± 0.62% | 37.77 ± 10.86% |
| Missing neurons | 0.30 | 43.72 ± 15.65% | 46.07 ± 17.63% | 70.52 ± 2.25% | 30.56 ± 10.08% |
| Missing neurons | 0.40 | 42.80 ± 14.74% | 45.13 ± 17.19% | 58.25 ± 1.89% | 24.98 ± 9.01% |
| Missing neurons | 0.50 | 42.08 ± 12.25% | 45.27 ± 15.08% | 49.59 ± 2.30% | 20.97 ± 6.95% |
| Distractors | 0.10 | 41.45 ± 16.08% | 47.73 ± 15.60% | 100.00 ± 0.00% | 41.45 ± 16.08% |
| Distractors | 0.20 | 38.37 ± 15.24% | 45.82 ± 16.02% | 100.00 ± 0.00% | 38.37 ± 15.24% |
| Distractors | 0.30 | 35.95 ± 16.43% | 43.06 ± 18.29% | 100.00 ± 0.00% | 35.95 ± 16.43% |
| Distractors | 0.40 | 31.53 ± 12.61% | 39.72 ± 14.94% | 100.00 ± 0.00% | 31.53 ± 12.61% |
| Distractors | 0.50 | 30.14 ± 13.98% | 38.55 ± 16.77% | 100.00 ± 0.00% | 30.14 ± 13.98% |

Missing conditions: Activity noise.
