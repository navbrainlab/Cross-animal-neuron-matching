# CPD robustness — native grouped CV5 × deterministic

Within each fold, perturbation replicates are averaged first; values are the unweighted mean ± sample SD across five folds. No shared-cohort rescoring is used.

| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| Clean | 0.00 | 26.51 ± 10.68% | 30.42 ± 11.38% | 100.00 ± 0.00% | 26.51 ± 10.68% |
| Coordinate noise | 0.02 | 26.12 ± 10.06% | 30.84 ± 11.09% | 100.00 ± 0.00% | 26.12 ± 10.06% |
| Coordinate noise | 0.05 | 24.18 ± 8.88% | 29.54 ± 12.14% | 100.00 ± 0.00% | 24.18 ± 8.88% |
| Coordinate noise | 0.10 | 21.11 ± 8.43% | 26.64 ± 12.22% | 100.00 ± 0.00% | 21.11 ± 8.43% |
| Coordinate noise | 0.20 | 16.89 ± 9.19% | 19.32 ± 9.66% | 100.00 ± 0.00% | 16.89 ± 9.19% |
| Missing neurons | 0.10 | 25.21 ± 10.56% | 29.53 ± 11.85% | 89.70 ± 1.63% | 22.61 ± 9.34% |
| Missing neurons | 0.20 | 25.49 ± 11.61% | 29.33 ± 11.67% | 79.88 ± 0.62% | 20.40 ± 9.24% |
| Missing neurons | 0.30 | 27.91 ± 12.93% | 30.20 ± 13.32% | 70.52 ± 2.25% | 19.49 ± 8.58% |
| Missing neurons | 0.40 | 24.86 ± 12.00% | 25.86 ± 12.65% | 58.25 ± 1.89% | 14.64 ± 7.30% |
| Missing neurons | 0.50 | 25.96 ± 9.72% | 26.56 ± 10.00% | 49.59 ± 2.30% | 12.99 ± 5.33% |
| Distractors | 0.10 | 26.88 ± 11.08% | 30.50 ± 11.47% | 100.00 ± 0.00% | 26.88 ± 11.08% |
| Distractors | 0.20 | 25.97 ± 10.64% | 30.45 ± 12.13% | 100.00 ± 0.00% | 25.97 ± 10.64% |
| Distractors | 0.30 | 26.36 ± 10.76% | 30.23 ± 11.57% | 100.00 ± 0.00% | 26.36 ± 10.76% |
| Distractors | 0.40 | 26.62 ± 9.93% | 31.52 ± 11.61% | 100.00 ± 0.00% | 26.62 ± 9.93% |
| Distractors | 0.50 | 25.65 ± 9.94% | 30.49 ± 11.39% | 100.00 ± 0.00% | 25.65 ± 9.94% |

Missing conditions: Activity noise.
