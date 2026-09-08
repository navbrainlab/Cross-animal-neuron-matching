# NuCLR official scratch-50k benchmark

This adapter is the primary NuCLR baseline for the manuscript benchmark.

## What is fixed to the official NuCLR calcium setup

- Pinned official source under `benchmark_official/third_party/nuclr_official`.
- `NuclrV2gaCa2` architecture.
- `SampleWiseContrastiveLoss`: tau=0.2, DCL=true, projector=true, full_denom=true.
- Same local neuron across two temporal views of the same recording is the positive pair.
- 30-second views; maximum temporal separation 240 seconds.
- UnitDropout minimum retained fraction 0.5.
- Batch size 16, BF16.
- AdamW with official decay/no-decay parameter grouping and weight decay 0.01.
- Max LR 1.25e-4.
- One-epoch linear warm-up then cosine decay.
- Gradient clipping at 1.0.
- 50,000 optimizer updates per fold/seed, following the official README guidance to set the epoch budget so total training steps are roughly 50k.

## Benchmark-specific adaptation

- Random initialization for every outer-fold/seed run; no EY/external checkpoint.
- Only outer-train worms participate in SSL training.
- Canonical cross-worm `cell_id` is never used in the training loss.
- Per-record sampling frequency is read from `sampling_rate_hz`, `source_fs`, or `fs` when present (fallback 4 Hz, matching the shared benchmark loader).
- Outer-validation is used only to choose a checkpoint along the fixed 50k-step training trajectory.
- Outer-test is not opened until training and validation-only selection are complete.
- Evaluation uses backbone embeddings, deterministic sequential 30-s windows, mean aggregation, L2-normalized cosine, all reference neurons, and the shared benchmark evaluator.

The only slight engineering instantiation of the official LR schedule is that cosine decay is parameterized directly by the exact 50,000-update budget. This is necessary because the NPZ-adapted official random window sampler can change the number of full batches by one between epochs. It preserves the official one-epoch warmup, maximum LR, and cosine-decay shape while preventing an LR rebound after the nominal horizon.

## Commands

From `/home/ubuntu/klb/nuclr/nuclr`:

```bash
python benchmark_official/adapters/nuclr_official_scratch50k/audit_plan.py
```

Engineering smoke (200 updates; never use as a paper result):

```bash
bash benchmark_official/adapters/nuclr_official_scratch50k/smoke.sh
```

Full 2 datasets x 5 folds x 3 seeds:

```bash
bash benchmark_official/adapters/nuclr_official_scratch50k/run_cv5x3.sh
```

The full runner is resumable through `checkpoints/last.pt` and reuses completed `outer_test/result.json` runs.
