# Zebrafish benchmark — LOFO8, one seed

All entries are the unweighted mean ± sample SD across the same eight held-out fish in the LOFO8 benchmark. Learned methods use the single fixed seed 42; CPD and Vanilla FGW are deterministic and therefore have no random-seed replicate. There is exactly one result per method and held-out-fish fold. Top-1 and Hungarian accuracy are reported in percent; MRR is unitless.

| Method | Top-1 ↑ | MRR ↑ | Hungarian Acc. ↑ |
|---|---:|---:|---:|
| CPD | 89.91 ± 3.92% | 0.9436 ± 0.0226 | 94.09 ± 3.47% |
| fDNC | 84.14 ± 11.76% | 0.9118 ± 0.0736 | 95.69 ± 7.98% |
| NuCLR | 2.77 ± 0.77% | 0.1008 ± 0.0186 | 2.80 ± 0.92% |
| GeoTransformer | 69.46 ± 8.00% | 0.7461 ± 0.0850 | 70.22 ± 10.19% |
| NGM-v2 | 86.20 ± 5.18% | 0.9174 ± 0.0336 | 86.20 ± 5.54% |
| Vanilla FGW | 88.14 ± 11.14% | 0.9360 ± 0.0643 | 87.84 ± 11.53% |
| FUGW (G+A; official) | 67.83 ± 8.60% | 0.8061 ± 0.0573 | 81.66 ± 8.69% |
| StatAtlas | 89.57 ± 17.84% | 0.9250 ± 0.1503 | 90.80 ± 18.07% |
| CRF_ID† | 96.64 ± 2.05% | 0.9819 ± 0.0118 | 96.76 ± 1.89% |
| **Ours** | **97.69 ± 3.84%** | **0.9867 ± 0.0238** | **97.62 ± 3.97%** |

Across the same eight held-out fish, NeuRID improves Top-1 over CRF_ID by
1.05 pp (paired 95% CI [-1.00, 3.09] pp; two-sided paired t interval). The
interval includes zero, so the observed mean advantage is not established as
a stable positive improvement across fish at the 95% confidence level.

## Audit note

The uncertainty is biological-fold variability (`n=8`, sample SD with `ddof=1`), not variation
across random seeds and not a bootstrap confidence interval.

For the NeuRID-versus-CRF_ID contrast, the eight held-out-fish differences are
paired before aggregation. Its 95% CI is
`mean(delta) +/- t(0.975, 7) * SD(delta) / sqrt(8)`; the fold-level inputs and
machine-readable result are in `query_evaluation/paired_top1_ours_vs_crfid.csv`
and `query_evaluation/paired_top1_ours_vs_crfid.json`.

The missing SDs were recovered from the eight held-out-fish values using the sample standard deviation (`ddof=1`):

- Vanilla FGW: `mrr_sd=0.0643387198`, `hungarian_sd=0.1152813611`.
- Ours: `mrr_sd=0.0238364028`, `hungarian_sd=0.0397484277`.

Primary source files:

- Ours: `zebrafish_mprt_lofo8_seed42_summary.json`.
- Vanilla FGW: `runs/zebrafish_vanilla_fgw_lofo8/fgw_lofo8_aggregate.json`.
- Official FUGW G+A: `runs/zebrafish_fugw_official_ga_lofo8/AUDIT_AND_SUMMARY.json`.
- GeoTransformer, NuCLR, and NGM-v2: their corresponding frozen aggregate files.
- StatAtlas: `statatlas_table1_core_lofo8_v1/aggregate.json`.
- CRF_ID†: `crfid_table1_core_lofo8_v1/aggregate.json`.

StatAtlas and CRF_ID† reuse their Table 1 core implementations. For each
physical q/r pair they build a common same-fish atlas from all remaining unique
timepoints, aligned by stable tracking ID; both endpoints are excluded. This is
a changed multi-reference input condition and must be provided equally to
compared methods. Both use positions only (`GG`). Endpoint IDs enter only the
metric calculation after the score matrices are complete.
