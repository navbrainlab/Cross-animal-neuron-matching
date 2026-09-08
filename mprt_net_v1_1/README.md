# MPRT-Net v0.1.1

Multimodal Population-Relational Transport for cross-animal neuron matching.

This is a from-scratch implementation for the unified Atanas (`000776`) and
RLD (`001623`) NPZ contracts.  It does not depend on the discarded model.

## v0.1.1 evaluation correction

The model architecture, forward pass, Sinkhorn solver, training loss, and
default `transport_steps=2` are unchanged from v0.1. The correction is limited
to evaluation and model selection:

- `Top1-real`, `Top5-real`, and `MRR-real` rank only real candidate neurons and
  are the benchmark metrics.
- Dustbin-inclusive metrics and the dustbin top-1 rate are reported separately.
- `best.pt` is selected only by validation `Top1-real`.
- Evaluation always calls `model.eval()`.
- Optional per-pair records and a paired cluster bootstrap comparison are
  included.

## What the model actually implements

1. Each animal is encoded independently by the same weights. Geometry and
   activity form node evidence and a dense, multi-scale soft relation field.
2. A relation-conditioned Population Encoder refines identity only with native
   within-animal context. There is no cross-animal attention in this stage.
3. Cross-animal unary similarity initializes an augmented, log-domain
   Sinkhorn plan.
4. The plan is refined by the exact contracted relational discrepancy

   \[
   D_{ij}(P)=\sum_{k,l}\bar P_{kl}
   \left\|U^A_{ik}-U^B_{jl}\right\|_2^2.
   \]

   The implementation expands the squared distance into norm and bilinear
   terms, requiring `O(d_r N^3)` compute and `O(d_r N^2)` memory rather than an
   explicit `O(N^4)` compatibility tensor.
5. Training uses one symmetric categorical objective on the final augmented
   plan. It does not add separate geometry and activity classification losses.

Unknown labels are ignored. They remain available as population context and
are **not** automatically assigned to the dustbin. Known unmatched examples
are created by synthetic node removal during training.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the equations and design boundaries.

## Install inside the existing repository

```bash
cd /path/to/NeuRID
tar -xzf mprt_net_v1_1.tar.gz
cd mprt_net_v1_1
conda activate nuclr310
python -m pip install -e . --no-deps
```

## Mandatory numerical and data check

Atanas:

```bash
python -m mprt_net.self_check \
  --dataset-root /path/to/NeuRID/Data/Atanas_SF_unified_000776/date_disjoint_v1/full \
  --split train \
  --device cuda
```

RLD:

```bash
python -m mprt_net.self_check \
  --dataset-root /path/to/NeuRID/Data/Dunn_001623/date_disjoint_full95_v1 \
  --split train \
  --device cuda
```

The check verifies the augmented marginals, the efficient relation contraction
against an exact brute-force calculation, real NPZ loading, end-to-end forward
and backward passes, nonzero relation gradients, and node-permutation
equivariance.

Before retraining, verify the corrected evaluator on the existing v0.1 Full
checkpoint:

```bash
python -m mprt_net.evaluate \
  --dataset-root /path/to/NeuRID/Data/Atanas_SF_unified_000776/date_disjoint_v1/full \
  --split val \
  --checkpoint /path/to/NeuRID/runs/mprt_v1/atanas/seed42/full/best.pt
```

For the checkpoint discussed during development, the expected invariant is
`queries=1898`, with `top1_real` around `0.7381` and
`top1_with_dustbin` around `0.7028`.

## Corrected Atanas seed-42 factorial

Run all four variants on two GPUs without overwriting the v0.1 pilot:

```bash
bash scripts/run_atanas_seed42_factorial.sh
```

This creates:

```text
/path/to/NeuRID/runs/mprt_v1_1/atanas/seed42/
  full/
  no_transport/
  no_population/
  node_only/
```

Each final training line is now explicitly:

```text
best_val_top1_real=... checkpoint=.../best.pt
```

After all four runs finish, compute query-paired and worm-pair-clustered
comparisons:

```bash
bash scripts/compare_atanas_seed42_factorial.sh
```

Each comparison reports Full and ablation Top-1, rescue/harm counts, pair
wins/ties/losses, a pair-cluster bootstrap, and the more conservative bootstrap
that resamples animals before reconstructing the induced pair graph.

## Individual seed-42 runs

Use the bundled launcher:

```bash
bash scripts/run_seed42.sh atanas 0
bash scripts/run_seed42.sh rld 0
```

Or run explicitly.

Atanas:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mprt_net.train \
  --dataset-root /path/to/NeuRID/Data/Atanas_SF_unified_000776/date_disjoint_v1/full \
  --output-dir /path/to/NeuRID/runs/mprt_v1_1/atanas/seed42/full \
  --variant full --seed 42 --epochs 80 --pairs-per-epoch 128
```

RLD:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mprt_net.train \
  --dataset-root /path/to/NeuRID/Data/Dunn_001623/date_disjoint_full95_v1 \
  --output-dir /path/to/NeuRID/runs/mprt_v1_1/rld/seed42/full \
  --variant full --seed 42 --epochs 80 --pairs-per-epoch 256
```

Evaluate the locked test split only after model choices are fixed on validation:

```bash
python -m mprt_net.evaluate \
  --dataset-root /path/to/NeuRID/Data/Atanas_SF_unified_000776/date_disjoint_v1/full \
  --split test \
  --checkpoint /path/to/NeuRID/runs/mprt_v1_1/atanas/seed42/full/best.pt \
  --output /path/to/NeuRID/runs/mprt_v1_1/atanas/seed42/full/test_metrics.json \
  --pair-output /path/to/NeuRID/runs/mprt_v1_1/atanas/seed42/full/test_pairs.jsonl
```

## First architecture ablations

The CLI exposes the minimum causal ablations needed before a larger sweep:

| Variant | What is removed |
| --- | --- |
| `full` | Nothing |
| `no_transport` | Cross-population relational refinement; retains native population formation |
| `no_population` | Relation-conditioned native population attention; retains relation transport |
| `node_only` | Both population attention and relational refinement |
| `geometry_only` | All activity node and edge inputs |
| `activity_only` | All geometry node and edge inputs |

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mprt_net.train \
  --dataset-root /path/to/NeuRID/Data/Atanas_SF_unified_000776/date_disjoint_v1/full \
  --output-dir /path/to/NeuRID/runs/mprt_v1_1/atanas/seed42/no_transport \
  --variant no_transport --seed 42 --epochs 80 --pairs-per-epoch 128
```

For the focal-loss control, rerun `full` with `--focal-gamma 0`. This is the
proper categorical cross-entropy comparison; `gamma=2` should not be assumed
better before validation.

## Dataset-specific note

The uploaded RLD NPZ contract contains no stimulus/epoch/window field. This
version therefore uses the full recording. Same-stimulus, non-stimulus, and
equal-length controls require a separate per-recording stimulus metadata file;
they should not be inferred from timestamps.
