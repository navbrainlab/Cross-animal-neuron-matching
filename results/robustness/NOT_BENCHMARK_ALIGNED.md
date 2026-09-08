# Historical output — not benchmark aligned

Do not report `FINAL_ROBUSTNESS_TABLES.md` or its figures as final results.

The 2026-08-31 fail-closed audit found:

- CPD and Ours use the shared `Data/Dunn_001623/cv5_grouped_v1` folds and reproduce their native
  seed-42 Benchmark cells.
- fDNC, NuCLR, and GeoTransformer cells in the current main Benchmark use a different 62/12/19
  fold family; their current-grouped robustness replays therefore cannot equal those Benchmark
  cells fold by fold.
- The final v2 table additionally rescored a shared common-query cohort, changing method-native
  denominators even for methods whose native Clean replay passed.
- Activity noise and distractor dustbin precision/recall/F1 are absent.
- Shared-CV5 seed-42 checkpoints exist for all learned methods, but shared-CV5 seeds 1 and 123
  currently exist only for Ours.

The compact release retains the authoritative gate as `AUDIT_GATE.json` in
this directory. The original large run directory and per-query outputs are not
part of the GitHub source package.

Required order of repair: unify the main Benchmark split and evaluator contract, freeze its exact
artifacts, pass every method/fold/seed severity-zero gate, add the missing corruption/metrics, and
only then evaluate nonzero corruptions without retuning.
