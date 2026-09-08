# Zebrafish benchmark — LOFO8, one seed

All entries are the unweighted mean ± sample SD across the same eight held-out fish in the LOFO8 benchmark. Learned methods use the single fixed seed 42; CPD, Euclidean, and Vanilla FGW are deterministic and therefore have no random-seed replicate. There is exactly one result per method and held-out-fish fold. Top-1 and Hungarian accuracy are reported in percent; MRR is unitless.

| Method | Top-1 ↑ | MRR ↑ | Hungarian Acc. ↑ |
|---|---:|---:|---:|
| CPD | 89.91 ± 3.92% | 0.9436 ± 0.0226 | 94.09 ± 3.47% |
| Euclidean | 75.29 ± 10.45% | 0.8572 ± 0.0693 | 92.43 ± 6.40% |
| fDNC | 84.14 ± 11.76% | 0.9118 ± 0.0736 | 95.69 ± 7.98% |
| NuCLR | 2.77 ± 0.77% | 0.1008 ± 0.0186 | 2.80 ± 0.92% |
| GeoTransformer | 69.46 ± 8.00% | 0.7461 ± 0.0850 | 70.22 ± 10.19% |
| RGM | 62.74 ± 13.50% | 0.7644 ± 0.0980 | 75.18 ± 12.29% |
| NGM-v2 | 86.20 ± 5.18% | 0.9174 ± 0.0336 | 86.20 ± 5.54% |
| Vanilla FGW | 88.14 ± 11.14% | 0.9360 ± 0.0643 | 87.84 ± 11.53% |
| **Ours** | **97.69 ± 3.84%** | **0.9867 ± 0.0238** | **97.62 ± 3.97%** |

## Audit note

The uncertainty is biological-fold variability (`n=8`, sample SD with `ddof=1`), not variation
across random seeds and not a bootstrap confidence interval.

The missing SDs were recovered from the eight held-out-fish values using the sample standard deviation (`ddof=1`):

- Euclidean: `mrr_sd=0.0692979834`, `hungarian_sd=0.0639783574`.
- Vanilla FGW: `mrr_sd=0.0643387198`, `hungarian_sd=0.1152813611`.
- Ours: `mrr_sd=0.0238364028`, `hungarian_sd=0.0397484277`.

Primary source files:

- Euclidean and Ours: `zebrafish_mprt_lofo8_seed42_summary.json` and `baselines/fold_{1..8}/euclidean.json`.
- Vanilla FGW: `runs/zebrafish_vanilla_fgw_lofo8/fgw_lofo8_aggregate.json`.
- GeoTransformer, NuCLR, RGM, and NGM-v2: their corresponding `runs/zebrafish_*_lofo8*/..._aggregate.json` files.
