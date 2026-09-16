# Prepared manuscript datasets

This directory contains only the three datasets used by the manuscript.

| Directory | Scientific source | Included form |
|---|---|---|
| `atanas/` | Atanas et al.; extracted from DANDI 000776 | 38 unique recording NPZ files plus five locked grouped folds |
| `kato_rld/` | Dunn/Kato-style RLD recordings; official processed release associated with DANDI 001623 and Zenodo 17353307 | 95 unique recording NPZ files plus five locked grouped folds |
| `zebrafish/` | Longitudinal embryonic spinal-cord recordings used by Wan et al. and Keller et al. | eight leave-one-fish-out folds with 1,616 pair-local NPZ records |

Atanas and Kato/RLD store each physical recording once under `recordings/`.
Fold directories use relative symbolic links, so cloning the repository keeps
the splits portable without duplicating binary blobs. Zebrafish records are
fold-specific and are stored directly in each fold.

Every NPZ follows `../neurid/DATA_CONTRACT.md`. The essential arrays are
`activity_raw`, `xyz`, and `cell_id`; masks and provenance metadata are also
retained. Split manifests never use test labels for training or checkpoint
selection.

Validate the binary assets with:

```bash
sha256sum -c data/SHA256SUMS
```

`SHA256SUMS` indexes physical NPZ files once; linked fold views are validated
by `scripts/check_release.py`.

Dataset terms are separate from the repository code license. Confirm the
upstream redistribution terms before making this repository public.
