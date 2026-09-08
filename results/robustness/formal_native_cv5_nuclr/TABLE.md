# NuCLR robustness — native grouped CV5 × seed42

Each perturbation seed is averaged within biological fold first; values are the unweighted mean ± sample SD across the same five folds as the Main Benchmark. No shared-cohort rescoring is used.

| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| Clean | 0.00 | 10.14 ± 7.62% | 13.16 ± 3.62% | 100.00 ± 0.00% | 10.14 ± 7.62% |
| Coordinate noise | 0.02 | 10.14 ± 7.62% | 13.16 ± 3.62% | 100.00 ± 0.00% | 10.14 ± 7.62% |
| Coordinate noise | 0.05 | 10.14 ± 7.62% | 13.16 ± 3.62% | 100.00 ± 0.00% | 10.14 ± 7.62% |
| Coordinate noise | 0.10 | 10.14 ± 7.62% | 13.16 ± 3.62% | 100.00 ± 0.00% | 10.14 ± 7.62% |
| Coordinate noise | 0.20 | 10.14 ± 7.62% | 13.16 ± 3.62% | 100.00 ± 0.00% | 10.14 ± 7.62% |
| Activity noise | 0.10 | 8.76 ± 4.95% | 10.91 ± 4.35% | 100.00 ± 0.00% | 8.76 ± 4.95% |
| Activity noise | 0.20 | 6.11 ± 3.40% | 6.84 ± 3.59% | 100.00 ± 0.00% | 6.11 ± 3.40% |
| Activity noise | 0.50 | 1.54 ± 2.36% | 4.05 ± 1.52% | 100.00 ± 0.00% | 1.54 ± 2.36% |
| Activity noise | 1.00 | 0.00 ± 0.00% | 3.20 ± 0.89% | 100.00 ± 0.00% | 0.00 ± 0.00% |
| Activity noise | 2.00 | 0.00 ± 0.00% | 2.19 ± 1.05% | 100.00 ± 0.00% | 0.00 ± 0.00% |
| Missing neurons | 0.10 | 9.84 ± 7.09% | 12.18 ± 3.04% | 89.70 ± 1.63% | 8.84 ± 6.45% |
| Missing neurons | 0.20 | 10.28 ± 7.52% | 14.27 ± 4.32% | 79.88 ± 0.62% | 8.20 ± 6.03% |
| Missing neurons | 0.30 | 10.94 ± 7.46% | 14.25 ± 3.26% | 70.52 ± 2.25% | 7.78 ± 5.31% |
| Missing neurons | 0.40 | 8.52 ± 6.67% | 11.71 ± 3.78% | 58.25 ± 1.89% | 4.86 ± 3.75% |
| Missing neurons | 0.50 | 10.21 ± 8.36% | 12.36 ± 5.59% | 49.59 ± 2.30% | 4.87 ± 3.90% |
| Distractors | 0.10 | 11.47 ± 8.16% | 12.98 ± 3.81% | 100.00 ± 0.00% | 11.47 ± 8.16% |
| Distractors | 0.20 | 11.35 ± 9.50% | 12.42 ± 4.27% | 100.00 ± 0.00% | 11.35 ± 9.50% |
| Distractors | 0.30 | 11.06 ± 9.62% | 10.82 ± 4.50% | 100.00 ± 0.00% | 11.06 ± 9.62% |
| Distractors | 0.40 | 10.56 ± 9.38% | 10.32 ± 4.34% | 100.00 ± 0.00% | 10.56 ± 9.38% |
| Distractors | 0.50 | 10.49 ± 8.73% | 8.95 ± 4.75% | 100.00 ± 0.00% | 10.49 ± 8.73% |

Coordinate noise leaves NuCLR unchanged because this baseline consumes activity only.
