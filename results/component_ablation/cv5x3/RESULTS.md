# NeuRID component ablation — locked test, 5 folds × 3 seeds

Values are the unweighted mean ± sample SD across 15 fold-seed cells (folds 0–4; seeds 1, 42, 123). Checkpoints were selected using validation data before the held-out test split was read.

## Atanas

| Variant | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian Acc. ↑ |
| --- | ---: | ---: | ---: | ---: |
| Full NeuRID | 74.24 ± 6.95% | 90.06 ± 5.43% | 0.8135 ± 0.0599 | 74.07 ± 6.98% |
| w/o Activity | 46.12 ± 6.74% | 80.62 ± 6.85% | 0.6110 ± 0.0645 | 46.20 ± 6.83% |
| Node-only | 61.87 ± 6.19% | 86.05 ± 6.07% | 0.7251 ± 0.0580 | 62.63 ± 6.45% |
| w/o Relation Transport | 73.53 ± 5.94% | 88.81 ± 4.75% | 0.8048 ± 0.0516 | 73.57 ± 6.01% |
| w/o Geometry (Activity-only) | 42.07 ± 4.01% | 75.80 ± 3.13% | 0.5728 ± 0.0341 | 40.56 ± 4.34% |

## RLD (Kato)

| Variant | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian Acc. ↑ |
| --- | ---: | ---: | ---: | ---: |
| Full NeuRID | 63.14 ± 4.45% | 80.10 ± 4.21% | 0.7091 ± 0.0388 | 61.22 ± 4.84% |
| w/o Activity | 47.40 ± 3.74% | 79.96 ± 2.91% | 0.6142 ± 0.0294 | 40.26 ± 3.02% |
| Node-only | 32.42 ± 4.80% | 60.11 ± 5.03% | 0.4525 ± 0.0387 | 30.52 ± 5.04% |
| w/o Relation Transport | 55.15 ± 4.61% | 73.66 ± 3.75% | 0.6390 ± 0.0366 | 54.21 ± 4.45% |
| w/o Geometry (Activity-only) | 21.10 ± 3.00% | 55.89 ± 4.68% | 0.3764 ± 0.0323 | 18.42 ± 3.12% |
