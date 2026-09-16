# NeuRID component ablation — formal CV5 × seed42

Values are the unweighted mean ± sample SD across the five biological folds. All four ablation arms were independently reconstructed from the archived per-query held-out-test results.

## Atanas

| Variant | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian ↑ |
|---|---:|---:|---:|---:|
| Full NeuRID | 74.92 ± 8.26% | 90.33 ± 5.69% | 0.8186 ± 0.0687 | 75.00 ± 8.37% |
| w/o Activity | 43.24 ± 6.23% | 78.93 ± 7.71% | 0.5881 ± 0.0636 | 43.87 ± 6.34% |
| w/o Population Relations | 62.93 ± 7.31% | 86.11 ± 6.72% | 0.7319 ± 0.0677 | 63.85 ± 8.03% |
| w/o Relation Transport | 73.16 ± 6.82% | 88.34 ± 5.31% | 0.8004 ± 0.0581 | 72.72 ± 6.99% |
| w/o Geometry | 41.07 ± 4.75% | 75.13 ± 4.04% | 0.5645 ± 0.0420 | 39.16 ± 4.31% |

## Kato/RLD

| Variant | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian ↑ |
|---|---:|---:|---:|---:|
| Full NeuRID | 63.93 ± 4.61% | 79.81 ± 5.01% | 0.7137 ± 0.0438 | 61.78 ± 5.18% |
| w/o Activity | 44.76 ± 4.07% | 79.81 ± 3.28% | 0.5991 ± 0.0346 | 37.49 ± 2.01% |
| w/o Population Relations | 33.95 ± 3.84% | 59.13 ± 4.34% | 0.4592 ± 0.0304 | 31.72 ± 4.86% |
| w/o Relation Transport | 55.91 ± 5.06% | 72.69 ± 3.15% | 0.6388 ± 0.0385 | 54.20 ± 5.20% |
| w/o Geometry | 20.76 ± 3.22% | 54.43 ± 2.02% | 0.3683 ± 0.0262 | 18.41 ± 3.71% |
