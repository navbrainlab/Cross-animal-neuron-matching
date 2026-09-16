# Unified covered-only main table — CV5 × seed42

Canonical test query: a neuron has valid, unique ground truth and its identity belongs to the union of unique supervised identities in that fold's outer-training split. The shared main-table cohort is the subset whose true identity is represented in the fold's frozen outer-training geometry medoid. Every method is scored on exactly that same cohort. Unlabeled/invalid neurons are excluded; identities outside the training union belong to a separate open-set evaluation.

All learned methods use model seed 42. Deterministic methods have no artificial seed. Values are the unweighted mean ± sample SD of five fold-level covered-only percentages. An em dash is fail-closed: at least one fold lacks an exactly rescorable shared-cohort artifact.

## ATANAS

| Method | Input modalities | Reference representation | Top-1 ↑ | Top-5 ↑ | Hungarian Accuracy ↑ |
| --- | --- | --- | ---: | ---: | ---: |
| CPD | Geometry | Single specimen | 48.57 ± 5.35% | 82.66 ± 4.48% | 51.90 ± 4.69% |
| fDNC | Geometry | Single specimen | 43.46 ± 7.02% | 83.19 ± 8.48% | 47.59 ± 8.36% |
| GeoTransformer | Geometry | Single specimen | 47.58 ± 3.88% | 71.94 ± 5.46% | 43.60 ± 4.16% |
| NGM-v2 | Geometry | Single specimen | 40.06 ± 2.82% | 77.80 ± 3.53% | 38.99 ± 2.50% |
| StatAtlas | Geometry | Statistical atlas | 5.15 ± 2.44% | 20.76 ± 7.63% | 7.89 ± 3.73% |
| CRF_ID | Geometry | Relational atlas | 24.94 ± 3.45% | 61.65 ± 6.53% | 24.94 ± 3.45% |
| NuCLR | Activity | Single specimen | 21.95 ± 1.81% | 47.46 ± 4.22% | 22.56 ± 3.03% |
| Vanilla FGW | Geometry + Activity | Single specimen | 25.17 ± 4.86% | 40.62 ± 5.37% | 23.53 ± 4.17% |
| FUGW (G+A; official) | Geometry + Activity | Single specimen | 32.37 ± 3.07% | 75.76 ± 6.34% | 37.20 ± 3.49% |
| NeurID (single medoid) | Geometry + Activity | Single specimen | 78.15 ± 8.46% | **94.20 ± 6.06%** | 75.78 ± 8.77% |
| NeurID (population atlas) | Geometry + Activity | Learned atlas | **79.01 ± 8.84%** | 92.78 ± 6.16% | **79.14 ± 8.94%** |

## Kato / RLD

| Method | Input modalities | Reference representation | Top-1 ↑ | Top-5 ↑ | Hungarian Accuracy ↑ |
| --- | --- | --- | ---: | ---: | ---: |
| CPD | Geometry | Single specimen | 26.51 ± 10.68% | 65.12 ± 27.02% | 30.42 ± 11.38% |
| fDNC | Geometry | Single specimen | 45.11 ± 16.99% | 88.64 ± 9.14% | 51.49 ± 18.65% |
| GeoTransformer | Geometry | Single specimen | 35.90 ± 9.82% | 52.98 ± 7.77% | 21.29 ± 6.83% |
| NGM-v2 | Geometry | Single specimen | 16.72 ± 6.55% | 64.16 ± 30.33% | 15.96 ± 5.71% |
| StatAtlas | Geometry | Statistical atlas | 7.67 ± 4.95% | 15.76 ± 5.33% | 3.73 ± 1.09% |
| CRF_ID | Geometry | Relational atlas | 1.70 ± 1.63% | 5.34 ± 4.89% | 1.70 ± 1.63% |
| NuCLR | Activity | Single specimen | 10.14 ± 7.62% | 47.09 ± 14.67% | 13.16 ± 3.62% |
| Vanilla FGW | Geometry + Activity | Single specimen | 6.32 ± 4.89% | 13.61 ± 4.61% | 5.02 ± 2.54% |
| FUGW (G+A; official) | Geometry + Activity | Single specimen | 25.18 ± 7.72% | 70.08 ± 22.84% | 8.55 ± 3.77% |
| NeurID (single medoid) | Geometry + Activity | Single specimen | **85.41 ± 6.19%** | **97.85 ± 1.34%** | **74.48 ± 5.64%** |
| NeurID (population atlas) | Geometry + Activity | Learned atlas | 74.78 ± 5.24% | 82.82 ± 5.68% | 73.30 ± 6.50% |

## Shared-cohort coverage diagnostic

| Dataset | Covered / full-canonical queries | Main-table covered-only Top-1 |
| --- | ---: | ---: |
| ATANAS | 72.46 ± 2.44% | 78.15 ± 8.46% |
| Kato / RLD | 34.08 ± 12.64% | 85.41 ± 6.19% |

Covered-only Top-1 is computed within each fold first, then summarized across folds; it is never the ratio of two five-fold means.

## Audit disposition


Hungarian values use each saved method's native assignment, but correctness is scored over the same medoid-covered query cohort. A stricter comparison that also forces an identical assignment domain requires new prediction-level reruns for every matcher and should be reported separately.

CRF_ID retains its fixed official 178-state relational atlas; the frozen medoid defines only the shared covered-query denominator.
