# Unified canonical-query main table — CV5 × seed42

Canonical test query: a neuron has valid, unique ground truth and its identity belongs to the union of unique supervised identities in that fold's outer-training split. Every method is scored on this complete canonical set. For a single-specimen method, a query whose true identity is absent from the frozen medoid reference is counted incorrect. Unlabeled/invalid neurons are excluded; identities outside the training union belong to a separate open-set evaluation.

All learned methods use model seed 42. Deterministic methods have no artificial seed. Values are the unweighted mean ± sample SD of five fold-level canonical-query percentages. An em dash is fail-closed: at least one fold lacks an exactly rescorable shared-cohort artifact.

## ATANAS

| Method | Input modalities | Reference representation | Top-1 ↑ | Top-5 ↑ | Hungarian Accuracy ↑ |
| --- | --- | --- | ---: | ---: | ---: |
| CPD | Geometry | Single specimen | 35.17 ± 3.71% | 59.94 ± 4.64% | 37.58 ± 3.23% |
| fDNC | Geometry | Single specimen | 31.48 ± 5.03% | 60.31 ± 6.69% | 34.48 ± 6.05% |
| GeoTransformer | Geometry | Single specimen | 34.52 ± 3.65% | 52.18 ± 4.99% | 31.63 ± 3.61% |
| NGM-v2 | Geometry | Single specimen | 29.00 ± 1.66% | 56.40 ± 3.71% | 28.24 ± 1.72% |
| StatAtlas | Geometry | Statistical atlas | 5.29 ± 2.02% | 21.14 ± 6.43% | 7.91 ± 3.41% |
| CRF_ID | Geometry | Relational atlas | 22.46 ± 3.41% | 58.08 ± 6.98% | 22.46 ± 3.41% |
| NuCLR | Activity | Single specimen | 15.93 ± 1.69% | 34.42 ± 3.71% | 16.35 ± 2.30% |
| Vanilla FGW | Geometry + Activity | Single specimen | 18.18 ± 3.21% | 29.41 ± 3.73% | 17.01 ± 2.77% |
| FUGW (G+A; official) | Geometry + Activity | Single specimen | 23.49 ± 2.76% | 54.94 ± 5.41% | 26.98 ± 2.91% |
| NeurID (single medoid) | Geometry + Activity | Single specimen | 56.70 ± 7.12% | 68.29 ± 5.45% | 54.98 ± 7.27% |
| NeurID (population atlas) | Geometry + Activity | Learned atlas | **74.92 ± 8.26%** | **90.33 ± 5.69%** | **75.00 ± 8.37%** |

## Kato / RLD

| Method | Input modalities | Reference representation | Top-1 ↑ | Top-5 ↑ | Hungarian Accuracy ↑ |
| --- | --- | --- | ---: | ---: | ---: |
| CPD | Geometry | Single specimen | 8.25 ± 2.53% | 19.60 ± 4.38% | 9.42 ± 2.37% |
| fDNC | Geometry | Single specimen | 14.23 ± 3.86% | 29.34 ± 8.11% | 15.92 ± 2.74% |
| GeoTransformer | Geometry | Single specimen | 11.37 ± 2.24% | 17.53 ± 5.29% | 6.88 ± 2.26% |
| NGM-v2 | Geometry | Single specimen | 5.21 ± 1.52% | 18.97 ± 5.46% | 4.99 ± 1.24% |
| StatAtlas | Geometry | Statistical atlas | 4.45 ± 2.08% | 13.22 ± 4.69% | 3.08 ± 1.93% |
| CRF_ID | Geometry | Relational atlas | 1.29 ± 0.98% | 5.64 ± 3.47% | 1.29 ± 0.98% |
| NuCLR | Activity | Single specimen | 3.85 ± 3.18% | 16.43 ± 7.89% | 4.78 ± 2.60% |
| Vanilla FGW | Geometry + Activity | Single specimen | 2.16 ± 1.52% | 4.48 ± 1.80% | 1.79 ± 1.11% |
| FUGW (G+A; official) | Geometry + Activity | Single specimen | 8.11 ± 2.86% | 21.88 ± 5.34% | 2.95 ± 1.76% |
| NeurID (single medoid) | Geometry + Activity | Single specimen | 29.01 ± 10.62% | 33.26 ± 12.03% | 25.45 ± 10.05% |
| NeurID (population atlas) | Geometry + Activity | Learned atlas | **63.93 ± 4.61%** | **79.81 ± 5.01%** | **61.78 ± 5.18%** |

## Single-medoid coverage diagnostic

| Dataset | Covered / full-canonical queries | Single-medoid overall Top-1 |
| --- | ---: | ---: |
| ATANAS | 72.46 ± 2.44% | 56.70 ± 7.12% |
| Kato / RLD | 34.08 ± 12.64% | 29.01 ± 10.62% |

Overall Top-1 is computed on all canonical queries within each fold first, then summarized across folds; it is never a pooled ratio across all folds.

Hungarian values use each saved method's native assignment, but correctness is scored over the same complete canonical query cohort. A stricter comparison that also forces an identical assignment domain requires new prediction-level reruns for every matcher and should be reported separately.

CRF_ID retains its fixed official 178-state relational atlas; it is scored on the same complete canonical query denominator.
