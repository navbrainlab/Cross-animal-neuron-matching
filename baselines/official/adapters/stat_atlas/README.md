# Statistical Atlas official locked-CV adapter

This adapter keeps the pinned third-party repository untouched.

## What is official
Atlas training calls the pinned official:
`models.Atlas.train_atlas(...)`

Official default training settings retained:
- min_counts = 2 (official condition is counts > 2)
- epsilon position = 1000
- n_iter = 10

## Position-only input
The official API expects color channels. The adapter supplies a constant one-dimensional
dummy color for every neuron. Therefore no RGB/activity information enters the atlas.

## Why test-time adaptation is necessary
Statistical Atlas is an atlas identity method, not a pairwise correspondence method.
For the locked neuron-correspondence benchmark we must produce an N x M score matrix
between two test worms.

The test adapter is strictly label-free:
1. PCA initialization.
2. Trimmed similarity ICP from test XYZ to atlas mean XYZ.
3. Equal-prior Gaussian posterior over official atlas identities using the learned
   position mu/sigma blocks.
4. Pairwise score = posterior overlap.

Outer-test labels are only accessed by the shared benchmark evaluator after the score
matrix has been completed.

## Smoke test
```bash
cd /home/ubuntu/klb/nuclr/nuclr
conda activate nuclr310

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -u baselines/official/adapters/stat_atlas/evaluate_statatlas_official.py \
  --dataset atanas --fold 1
```

## Full run
```bash
cd /home/ubuntu/klb/nuclr/nuclr
conda activate nuclr310

bash baselines/official/adapters/stat_atlas/run_statatlas_official_cv5.sh
```

Outputs:
- `baselines/official/runs/stat_atlas_official/query_level.csv`
- `baselines/official/runs/stat_atlas_official/fold_metrics.csv`
- `baselines/official/runs/stat_atlas_official/summary.csv`
- fold-specific trained atlas / metrics / diagnostics
- `PROVENANCE.json`


## Exact position-only adaptation

The pinned official `Atlas.update_beta` consists of independent position and
color branches. This adapter subclasses `Atlas` and overrides only
`update_beta`:

- keeps the official XYZ `MCR_solver` regression,
- keeps the official XYZ Mahalanobis cost,
- disables color regression,
- disables color Mahalanobis cost,
- carries a constant dummy color dimension through unchanged only because the
  rest of `train_atlas()` expects a `(3+C)` array.

All other `train_atlas()` steps are inherited from the pinned official class:
initialization, `estimate_mu`, `estimate_sigma`, iteration schedule, and final
atlas construction.

No pseudoinverse fallback and no random dummy color are used.


## Degenerate-geometry handling for RLD

Some RLD training worms have effectively low-dimensional position geometry.
The pinned official `scaled_rotation` standardizes coordinates dimension-wise;
a zero-variance dimension can therefore produce NaNs before SVD.

The adapter uses the official path whenever it returns finite values. Only for
a degenerate training alignment does it fall back to an isotropic
least-squares similarity fit on the already identity-aligned TRAIN rows.
Likewise, if the XYZ MCR normal equations are singular, the same official
`MCR_solver` is retried with a Moore-Penrose pseudoinverse for that call.

These are train-only numerical fallbacks. No outer-test labels are used.
Fallback counts are saved in each fold's diagnostics.
