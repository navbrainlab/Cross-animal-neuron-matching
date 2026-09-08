# Complete NeuRID model code

The complete model implementation is in `neurid/mprt_net/`. The old
NuCLR files under `src/` are baseline dependencies and are not the NeuRID
model.

## Core implementation

| File | Role |
|---|---|
| `neurid/mprt_net/model.py` | `NeuRID` public alias and checkpoint-compatible `MPRTNet` implementation |
| `neurid/mprt_net/config.py` | Architecture configuration and ablation switches |
| `neurid/mprt_net/layers.py` | Geometry/activity encoders and relation-conditioned population layers |
| `neurid/mprt_net/relations.py` | Multimodal within-animal relation construction and normalization |
| `neurid/mprt_net/sinkhorn.py` | Capacity-aware dustbin and log-domain relational transport |
| `neurid/mprt_net/losses.py` | Symmetric focal matching and optional cycle losses |
| `neurid/mprt_net/data.py` | NPZ contract, caching, pair construction, target masks, activity resampling |
| `neurid/mprt_net/metrics.py` | Top-k, MRR, Hungarian and dustbin metrics |
| `neurid/mprt_net/train.py` | Pairwise training entry point |
| `neurid/mprt_net/build_anchored_atlas.py` | Learned identity-anchored atlas construction |
| `neurid/mprt_net/evaluate.py` | Pairwise and learned-atlas evaluation |
| `neurid/mprt_net/self_check.py` | Numerical, gradient, data-contract and permutation checks |
| `neurid/mprt_net/experiments/` | Paired evaluation, complexity and robustness utilities |

`neurid/ARCHITECTURE.md` gives the equations and
`neurid/DATA_CONTRACT.md` specifies the input files.

## Model used for the formal Ours result

The reported model uses geometry and activity jointly, relation-conditioned
within-animal population encoding, cross-population relation transport, a
capacity-correct dustbin, and a learned identity-anchored atlas. The static
atlas is built from outer-training animals only. Checkpoint selection uses the
validation split; held-out test labels are evaluation-only.

The formal main result is exactly five grouped biological folds with the
single model seed `42`. Its fold membership and admission gate are recorded in
`results/main_benchmark_seed42/PROTOCOL.json`, `artifact_audit.csv`, and
`readiness.json`. Training uses `mprt_net.train`, the train-only identity atlas
uses `mprt_net.build_anchored_atlas`, and evaluation uses the static-atlas
evaluator under `scripts/neurid/`; the formal aggregator is
`scripts/benchmarks/summarize_unified_cv5_seed42.py`.

Files whose names contain `cv5x3`, including
`scripts/neurid/run_final_cv5x3_s1_42_123.py` and the CV5 × 3-seed component
ablation runner, are retained only to reproduce historical/supplementary
stability analyses. Their three-seed aggregates are not formal results. The
seed-42 component-ablation fold cells are separately admitted by the formal
protocol under `results/component_ablation/seed42/`.

## Minimal installation and checks

```bash
python -m pip install -e './neurid[eval,test]'
python -m pytest -q neurid/tests
python -m mprt_net.self_check \
  --dataset-root /path/to/fold_or_dataset \
  --split train \
  --device cpu
```

The repository does not include raw datasets or trained weights. Dataset
preparation and frozen split manifests are under `workflows/`; result tables
and fold-level records are under `results/`.
