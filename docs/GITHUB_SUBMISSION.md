# GitHub submission scope

This source release is intentionally a code-and-protocol submission. It does
not include local datasets, trained weights, raw per-query predictions, large
paper intermediates, logs, caches, or nested third-party Git repositories.

## Included

- Primary implementation: `mprt_net_v1_1/`.
- Current experiment entry points: `scripts/benchmarks/`,
  `scripts/fair_identity/`, `scripts/lib/`, `scripts/mprt/`,
  `scripts/mechanisms/`, `scripts/robustness/`, `scripts/scaling/`, and
  `scripts/zebrafish/`.
- The standalone official-fDNC checkpoint loader retained in `engines/`;
  pair scoring lives in `scripts/lib/fdnc_scoring.py` to avoid importing the
  historical model-development stack.
- Locked Atanas and Kato/RLD workflow code and split manifests in `workflows/`.
- Relevant baseline adapters, manifests, and protocol definitions in
  `benchmark_official/`, excluding its `envs`, `logs`, `results`, `runs`,
  `third_party`, and `work` directories.
- Historical/supplementary CV5 × 3-seed provenance in
  `paper_submission_data/5fold_cross_validation/`; it is not a formal-result
  source. Formal Atanas and Kato/RLD main results use five folds and seed 42
  only; zebrafish uses its separate LOFO8 protocol.
- Compact result records in `results/`: protocol locks, audits, fold-level
  metrics, summary tables and publication figures for every experiment listed
  in the release result index. Large raw predictions remain excluded.
- The audited Atanas GWOT-MD adaptation in `gwot_md_atanas21_paper/`, without
  its local dataset snapshots or generated HDF5 files.
- Project-specific GeoTransformer, RGM, and NGM-v2 overlays in
  `third_party_adapters/`.
- Repository documentation and benchmark protocol files.

## Excluded

- `Data/`, `runs/`, `logs/`, `CURRENT_RESULTS/`, `table1_clean/`, and `archive/`.
- Checkpoints and serialized arrays (`*.pt`, `*.ckpt`, `*.pkl`, and raw data).
- Nested repositories under `CRF_Cell_ID/`, `third_party/`, and
  `benchmark_official/third_party/`.
- Paper archives and large raw mechanism tables.
- Cross-domain, Chaudhary, and unrelated-dataset code archived on 2026-09-04.
  Scaling code is included in this release because it supports reported
  runtime and training-population experiments.

## Pre-push checks

```bash
python scripts/check_release.py
python -m pytest -q -p no:cacheprovider mprt_net_v1_1/tests
find mprt_net_v1_1/scripts scripts -type f -name '*.sh' -print0 \
  | xargs -0 -n1 bash -n
git diff --cached --check
git diff --cached --stat
```

Do not use `git add -A` in the current working tree: it contains unrelated
pre-existing deletions from the upstream NuCLR checkout. Stage only the paths
listed in the release manifest.

Some provenance-oriented baseline and scaling scripts still contain their original
absolute workspace paths. The primary `mprt_net_v1_1` launchers are portable;
legacy baseline paths must be replaced with local dataset and upstream-checkout
locations before execution.
