# Component ablation — main clean-benchmark protocol

Protocol: the exact grouped five-fold test split and model seed 42 used by the main clean benchmark. Each value is the unweighted mean ± sample SD across five biological folds. The Full arm passed a fold-by-fold reproduction gate against the current Ours/static benchmark cells.

## ATANAS

| Arm | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian ↑ | ΔTop-1 vs Full |
|---|---:|---:|---:|---:|---:|
| Full MPRT-Net | 74.92 ± 8.26% | 90.33 ± 5.69% | 0.8186 ± 0.0687 | 75.00 ± 8.37% | — |
| w/o Activity | 43.24 ± 6.23% | 78.93 ± 7.71% | 0.5881 ± 0.0636 | 43.87 ± 6.34% | -31.67 ± 6.90 pp |
| w/o Population Relations | 62.93 ± 7.31% | 86.11 ± 6.72% | 0.7319 ± 0.0677 | 63.85 ± 8.03% | -11.98 ± 3.42 pp |
| w/o Relation Transport | 73.16 ± 6.82% | 88.34 ± 5.31% | 0.8004 ± 0.0581 | 72.72 ± 6.99% | -1.75 ± 2.15 pp |

## RLD

| Arm | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian ↑ | ΔTop-1 vs Full |
|---|---:|---:|---:|---:|---:|
| Full MPRT-Net | 63.93 ± 4.61% | 79.81 ± 5.01% | 0.7137 ± 0.0438 | 61.78 ± 5.18% | — |
| w/o Activity | 44.76 ± 4.07% | 79.81 ± 3.28% | 0.5991 ± 0.0346 | 37.49 ± 2.01% | -19.18 ± 3.88 pp |
| w/o Population Relations | 33.95 ± 3.84% | 59.13 ± 4.34% | 0.4592 ± 0.0304 | 31.72 ± 4.86% | -29.99 ± 3.14 pp |
| w/o Relation Transport | 55.91 ± 5.06% | 72.69 ± 3.15% | 0.6388 ± 0.0385 | 54.20 ± 5.20% | -8.02 ± 4.18 pp |
