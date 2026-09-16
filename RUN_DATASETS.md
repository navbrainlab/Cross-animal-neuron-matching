# Included datasets and commands

All paths below are relative to the repository root.

| Dataset | Prepared root | Protocol |
|---|---|---|
| Atanas | `data/atanas` | five locked grouped folds, seed 42 |
| Kato/RLD | `data/kato_rld` | five locked grouped folds, seed 42 |
| Zebrafish | `data/zebrafish` | eight leave-one-fish-out folds, seed 42 |

Install the package once:

```bash
python -m pip install -e './neurid[test]'
```

Run Atanas and Kato/RLD:

```bash
python scripts/neurid/run_cv5.py --datasets atanas,kato_rld --device cuda
```

Run one fold through the complete train → train-only atlas → test pipeline:

```bash
python scripts/neurid/run_cv5.py \
  --datasets atanas --folds 0 --device cpu
```

The zebrafish files are already organized as `fold_1` through `fold_8`, each
with `train`, `val`, and `test` directories and a frozen protocol JSON. The
scripts under `scripts/zebrafish/` implement the paper's pair-local protocol;
results must be aggregated by held-out fish, not by pooling queries across
folds.

Validate every included binary file before running:

```bash
sha256sum -c data/SHA256SUMS
```
