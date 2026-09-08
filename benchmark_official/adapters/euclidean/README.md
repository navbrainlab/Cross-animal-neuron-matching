# Euclidean locked benchmark

Deterministic position-only baseline.

Protocol:
- exact locked biological test folds from Candidate B;
- seed routes 1/42/123 are audited to contain identical test worms;
- one deterministic evaluation per biological fold;
- worm-wise normalized XYZ already exposed by the locked Candidate-B loader;
- pair score: negative squared Euclidean distance;
- no labels are used for alignment or score construction;
- test labels are used only by the shared final evaluator;
- Direct Top-1, Top-3, Top-5, MRR, and Hungarian accuracy use
  `benchmark_cv5x3_common.evaluate_pairwise`.

Statistics:
- unweighted mean ± sample SD across the five biological outer folds;
- no seed dimension because Euclidean is deterministic.
