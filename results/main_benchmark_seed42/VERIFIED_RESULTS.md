# Unified CV5 × seed42 benchmark — verified cells only

Formal protocol: five locked biological folds and the single learned-model
seed `42`; mean ± sample SD is computed over the five fold-level values.
Deterministic methods have no artificial seed. A method appears only after all
five folds pass `artifact_audit.csv`. This is a fail-closed table and remains
incomplete only for the groups listed under “Excluded pending
rerun/provenance”.

## ATANAS

| Method | Top-1 | Top-5 | Hungarian Accuracy |
| --- | ---: | ---: | ---: |
| CPD | 48.57 ± 5.35% | 82.66 ± 4.48% | 51.90 ± 4.69% |
| GeoTransformer | 44.37 ± 4.33% | 63.30 ± 5.88% | 40.89 ± 3.09% |
| NGM-v2 | 41.68 ± 5.53% | 81.25 ± 6.14% | 41.72 ± 6.13% |
| NuCLR | 21.76 ± 1.81% | 47.23 ± 4.24% | 22.37 ± 3.04% |
| Ours | 74.92 ± 8.26% | 90.33 ± 5.69% | 75.00 ± 8.37% |
| RGM | 37.86 ± 5.26% | 73.82 ± 8.88% | 34.12 ± 5.23% |
| StatAtlas | 5.26 ± 2.02% | 21.02 ± 6.43% | 7.87 ± 3.42% |
| Vanilla FGW | 20.49 ± 4.97% | 37.55 ± 8.71% | 20.45 ± 4.93% |
| fDNC | 43.46 ± 7.02% | 83.19 ± 8.48% | 47.59 ± 8.36% |

## RLD

| Method | Top-1 | Top-5 | Hungarian Accuracy |
| --- | ---: | ---: | ---: |
| CPD | 26.51 ± 10.68% | 65.12 ± 27.02% | 30.42 ± 11.38% |
| GeoTransformer | 16.52 ± 13.33% | 35.70 ± 19.10% | 9.29 ± 6.98% |
| NGM-v2 | 8.48 ± 2.46% | 35.36 ± 8.45% | 6.73 ± 3.50% |
| NuCLR | 10.14 ± 7.62% | 47.09 ± 14.67% | 13.16 ± 3.62% |
| Ours | 63.93 ± 4.61% | 79.81 ± 5.01% | 61.78 ± 5.18% |
| RGM | 8.92 ± 6.32% | 25.08 ± 12.89% | 5.08 ± 2.63% |
| StatAtlas | 4.26 ± 2.04% | 12.70 ± 4.71% | 2.95 ± 1.87% |
| Vanilla FGW | 7.99 ± 2.44% | 13.27 ± 2.10% | 7.85 ± 2.52% |
| fDNC | 45.11 ± 16.99% | 88.64 ± 9.14% | 51.49 ± 18.65% |

## Excluded pending rerun/provenance

- atanas / CRF_ID: MISSING, MISSING, MISSING, MISSING, MISSING
- atanas / GWOT-MD: MISSING, MISSING, MISSING, MISSING, MISSING
- atanas / GWOT-MD (our adaptation): MISSING, MISSING, MISSING, MISSING, MISSING
- rld / CRF_ID: MISSING, MISSING, MISSING, MISSING, MISSING
- rld / GWOT-MD: MISSING, MISSING, MISSING, MISSING, MISSING
- rld / GWOT-MD (our adaptation): MISSING, MISSING, MISSING, MISSING, MISSING
