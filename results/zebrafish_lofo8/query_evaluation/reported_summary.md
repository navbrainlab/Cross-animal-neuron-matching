# Zebrafish LOFO8 reported results

Unweighted mean ± sample SD (`ddof=1`) across the eight archived fold records.

| Method | Top-1 | Top-5 | MRR | Hungarian |
|---|---:|---:|---:|---:|
| CPD | 89.91 ± 3.92% | 99.74 ± 0.38% | 0.9436 ± 0.0226 | 94.09 ± 3.47% |
| fDNC | 84.14 ± 11.76% | 99.80 ± 0.56% | 0.9118 ± 0.0736 | 95.69 ± 7.98% |
| NuCLR | 2.77 ± 0.77% | 12.43 ± 3.06% | 0.1008 ± 0.0186 | 2.80 ± 0.92% |
| GeoTransformer | 69.46 ± 8.00% | 80.87 ± 9.63% | 0.7461 ± 0.0850 | 70.22 ± 10.19% |
| NGM-v2 | 86.20 ± 5.18% | 99.11 ± 1.07% | 0.9174 ± 0.0336 | 86.20 ± 5.54% |
| Vanilla FGW | 88.14 ± 11.14% | 99.99 ± 0.02% | 0.9360 ± 0.0643 | 87.84 ± 11.53% |
| FUGW (G+A; official) | 67.83 ± 8.60% | 97.09 ± 1.64% | 0.8061 ± 0.0573 | 81.66 ± 8.69% |
| StatAtlas | 89.57 ± 17.84% | 95.62 ± 12.35% | 0.9250 ± 0.1503 | 90.80 ± 18.07% |
| CRF_ID† | 96.64 ± 2.05% | 99.93 ± 0.12% | 0.9819 ± 0.0118 | 96.76 ± 1.89% |
| Ours | 97.69 ± 3.84% | 99.90 ± 0.27% | 0.9867 ± 0.0238 | 97.62 ± 3.97% |

## Paired Top-1 comparison

For each held-out fish, the difference is `NeuRID Top-1 - CRF_ID Top-1`. The confidence interval is a two-sided paired t interval over the eight fish-level differences.

NeuRID improves Top-1 over CRF_ID by 1.05 pp (paired 95% CI [-1.00, 3.09] pp).
Because the interval includes zero, these eight fish do not establish a stable positive improvement at the 95% confidence level.
