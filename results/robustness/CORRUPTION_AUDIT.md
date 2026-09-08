# RLD corruption audit v2

Status: **PASSED — READY FOR FINAL FIGURE** (2026-08-26)

## Locked paired inputs

- Split: `Data/Dunn_001623/cv5_grouped_v1`, folds 0–4, 19 held-out worms per fold.
- Model seed/checkpoints: seed 42 current-grouped checkpoints; no retraining or test-time checkpoint selection.
- Shared corruption manifest SHA-256: `313f9b4fee0be98af9e8a4303c81ea32941ddfda570675ad7987bc87feec8271`.
- Manifest audit: passed, 225 conditions and 4,275 materialized test files checked.
- Every method receives the same deleted row indices, coordinate perturbations, distractor rows, and manifest query rows.
- Distractors have `valid_xyz_mask=true` and all supervision masks false, so they enter inference but cannot become scored ground truth.

## Common estimand

The formal paired result uses the exact manifest query cohort for every method. A label that a method cannot represent stays in the denominator and scores zero. Top-1 is conditional on surviving common queries; coverage is surviving/common-clean queries; effective Top-1 is correct/common-clean queries; retention is effective Top-1 divided by clean Top-1.

Aggregation is an unweighted corruption-draw mean within each biological fold followed by an unweighted mean over five folds. All intervals use 10,000 paired hierarchical bootstrap replicates (folds, worms within fold, and shared draw IDs; seed 20260826). The same bootstrap indices are used for every method and pairwise contrast.

## Completed methods

CPD, fDNC, NuCLR, GeoTransformer, and Ours static Atlas each completed 225/225 conditions. All native severity-zero replays passed the saved current-grouped seed-42 clean guards. Native clean Top-1 values are CPD 26.51%, fDNC 45.11%, NuCLR 10.14%, GeoTransformer 16.52%, and Ours static Atlas 63.93%. These native values are audit checks, not the formal common-cohort scores.

Common-cohort clean Top-1 (95% CI): CPD 7.86% (4.77–11.39), fDNC 13.68% (10.03–17.74), NuCLR 3.66% (1.20–6.58), GeoTransformer 6.03% (1.96–10.26), and Ours 61.10% (54.08–67.86). Coverage is numerically identical across all five methods in every one of the 17 plotted corruption cells (maximum cross-method spread 0.0).

At maximum severity, common-cohort results are:

| Corruption | Method | Top-1 | Effective Top-1 | Retention |
|---|---:|---:|---:|---:|
| Coordinate noise 0.20 | CPD | 4.69% | 4.69% | 59.74% |
| Coordinate noise 0.20 | fDNC | 7.05% | 7.05% | 51.53% |
| Coordinate noise 0.20 | NuCLR | 3.66% | 3.66% | 100.00% |
| Coordinate noise 0.20 | GeoTransformer | 1.97% | 1.97% | 32.62% |
| Coordinate noise 0.20 | Ours | 30.78% | 30.78% | 50.37% |
| Missing 0.50 | CPD | 7.82% | 3.86% | 49.17% |
| Missing 0.50 | fDNC | 12.98% | 6.39% | 46.73% |
| Missing 0.50 | NuCLR | 3.57% | 1.74% | 47.62% |
| Missing 0.50 | GeoTransformer | 5.31% | 2.62% | 43.48% |
| Missing 0.50 | Ours | 53.24% | 26.17% | 42.84% |
| Distractors 0.50 | CPD | 7.64% | 7.64% | 97.22% |
| Distractors 0.50 | fDNC | 9.06% | 9.06% | 66.19% |
| Distractors 0.50 | NuCLR | 3.69% | 3.69% | 100.60% |
| Distractors 0.50 | GeoTransformer | 3.14% | 3.14% | 52.18% |
| Distractors 0.50 | Ours | 54.27% | 54.27% | 88.82% |

## Distractor audit

For every method, 0/75 nonzero distractor fold/draw cells are bitwise identical to their fold clean metrics. This rejects the old failure mode in which CPD and fDNC distractor curves were unchanged because inserted rows were not valid model inputs. CPD's relative effect remains small and its retention CI includes no-effect at several levels, but its raw outputs are no longer invariant.

## Final gate

All five methods completed all conditions, passed their native clean guards, and entered the same common-cohort 10,000-replicate paired bootstrap. The final paper figure is allowed.

## Caption

“Robustness on RLD under coordinate noise, missing neurons, and distractor neurons. Results use the fixed current-grouped five-fold split (19 held-out worms per fold) and each method's fold-specific seed-42 checkpoint selected on clean validation data only. CPD has no learned checkpoint; fDNC uses `runs/fdnc_current_grouped_cv_v2/rld/fold{f}/seed42/selected/best.pt`; NuCLR uses `benchmark_official/runs/nuclr_official_scratch50k_current_cv_seed42/rld/fold_{f}/seed_42/selected/best.pt`; GeoTransformer uses its fold-specific selected `iter-*.pth.tar`; Ours uses `runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld/fold{f}/seed42/dynamic/low_rank_r8/best.pt`. All methods replay the same precomputed corruption manifest (`313f9b4f…8271`) and the same manifest query cohort; unrepresentable labels remain in the denominator and score zero. Curves show conditional Top-1 accuracy computed over matchable neurons, averaged over corruption draws within each biological fold and then equally over five folds; shading denotes paired 95% hierarchical-bootstrap intervals over folds, worms, and shared corruption draws (10,000 replicates). Coordinate noise σ is the Gaussian coordinate-noise standard deviation relative to the within-worm scale. Missing-neuron GT coverage decreases from 100% to 49.2%; Effective Top-1 is reported in the complete tables. Severity-zero native replays were separately required to reproduce each saved clean benchmark before corrupted conditions were accepted.”
