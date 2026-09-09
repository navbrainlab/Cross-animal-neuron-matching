# StatAtlas / CRF-ID zebrafish LOFO8

This directory contains the complete compact record for the position-only
zebrafish comparison: 16 validation locks, 16 fold-level test results,
per-pair sufficient statistics, and the eight-fish aggregate.

Run from the repository root:

```bash
PYTHONPATH=neurid:. python scripts/zebrafish/evaluate_zebrafish_statatlas_crfid_lofo8.py select --data-root /path/to/Zebrafish_MPRT_LOFO8_60m --run-root runs/zebrafish_statatlas_crfid_pairwise_lofo8_v1
PYTHONPATH=neurid:. python scripts/zebrafish/evaluate_zebrafish_statatlas_crfid_lofo8.py test --data-root /path/to/Zebrafish_MPRT_LOFO8_60m --run-root runs/zebrafish_statatlas_crfid_pairwise_lofo8_v1
PYTHONPATH=neurid:. python scripts/zebrafish/evaluate_zebrafish_statatlas_crfid_lofo8.py aggregate --run-root runs/zebrafish_statatlas_crfid_pairwise_lofo8_v1
```

Important: these are pairwise adaptations, not canonical global-identity atlas
runs. Zebrafish identities are local to each longitudinal pair. Test identities
are used only for metrics after matching scores are complete.
