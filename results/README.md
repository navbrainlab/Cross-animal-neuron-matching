# Experiment results and code map

This directory is the compact, GitHub-safe record for the experiments listed
in `USER_PROVIDED_EXPERIMENTS.md`. Raw datasets, checkpoints, large per-query
matrices and caches are intentionally excluded. Fold-level metrics, protocol
locks, audits, summary tables and publication figures are retained.

The formal Atanas and Kato/RLD main-result definition is **five biological
folds × the single model seed 42**. The uncertainty is the sample SD across
five fold values, not across seeds. Deterministic methods do not receive an
artificial seed. Every three-seed table is explicitly supplementary or
historical. Zebrafish remains a separately defined LOFO8 × seed-42 experiment.

## Experiment index

| Experiment | Result records | Reproduction code | Status |
|---|---|---|---|
| Atanas and Kato/RLD main benchmark, CV5 × seed42 | `main_benchmark_seed42/` | `../scripts/benchmarks/`, `../scripts/fair_identity/`, `../baselines/adapters/` | Canonical; 5/5 PASS for every populated row |
| Single-specimen train-medoid protocol, historical CV5 × 3 seeds | `single_specimen_medoid/` | `../scripts/lib/fair_identity_protocol.py`, `../scripts/fair_identity/` | Audited medoid selection; cross-method fold families differ, supplementary only |
| NeuRID component ablation, CV5 × seed42 | `component_ablation/seed42/` | `../scripts/neurid/run_mprt_component_ablation_cv5x3.py`, `../scripts/neurid/summarize_component_ablation_seed42.py` | Main-protocol aligned |
| NeuRID component ablation, CV5 × 3 seeds | `component_ablation/cv5x3/` | same runner and merge scripts | Historical/supplementary stability |
| Zebrafish LOFO8 | `zebrafish_lofo8/` | `../scripts/zebrafish/`, baseline overlays under `../baselines/adapters/` | Complete eight-fold table |
| Zebrafish StatAtlas/CRF-ID pairwise adaptations | `zebrafish_lofo8/statatlas_crfid_pairwise_lofo8_v1/` | `../scripts/zebrafish/evaluate_zebrafish_statatlas_crfid_lofo8.py` | Complete: 16 validation locks and 16 held-out fold-method results |
| Cross-animal population relations | `mechanisms/` | `../scripts/mechanisms/` | Compact tables and figures retained |
| Coordinate noise, missing neurons and distractors | `robustness/formal_native_cv5_*` | `../scripts/robustness/` | Use only the `formal_*` outputs |
| Activity noise | `robustness/formal_activity_noise_v1/` | robustness preparation/evaluation/summarization scripts | Main-protocol severity-zero gate passed |
| Unknown rejection and no-dustbin ablation | `robustness/formal_dustbin_ablation_ours/` | `../scripts/robustness/eval_ours_rld_dustbin_ablation_current_grouped.py` | Main-protocol aligned |
| GPU runtime scaling | `scaling/runtime/` | `../scripts/scaling/benchmark_full_wallclock_runtime_gpu1.py` | RTX 4090, 100 repeats; full online pipeline timing |
| Training-population scaling | `scaling/training_population/` | `../scripts/scaling/run_rld_training_population_scaling_v2.py` | Independent fixed-test hierarchical-bootstrap analysis |

## Primary code entry points

- Complete model, training, atlas building, and evaluation:
  `../neurid/mprt_net/model.py`, `train.py`,
  `build_anchored_atlas.py`, and `evaluate.py`.
- Copy-and-run single-fold and formal CV5 × seed42 instructions:
  `../RUN_MODEL.md`; the one-command runner is
  `../scripts/neurid/run_model.py`, and generic five-fold aggregation is
  implemented by `../scripts/neurid/summarize_cv5_seed42.py`.
- Formal result admission and aggregation:
  `../scripts/benchmarks/audit_unified_cv5_seed42.py` and
  `summarize_unified_cv5_seed42.py`.
- Single-specimen train-medoid evaluation:
  `../scripts/fair_identity/evaluate_train_reference_ensemble.py` and
  `summarize_medoid_template_cv5x3.py`.
- Full model and component ablation runs:
  `../scripts/neurid/run_final_cv5x3_s1_42_123.py`,
  `run_mprt_component_ablation_cv5x3.py`, and the two ablation summarizers.
- Population-relation mechanism analysis:
  `../scripts/mechanisms/analyze_cross_animal_population_relation_mechanism.py`
  plus the plotting scripts in that directory.
- Robustness preparation, method-specific replay, audits, and summaries:
  all scripts in `../scripts/robustness/`; filenames identify each method and
  corruption family.
- Runtime and training-population scaling:
  `../scripts/scaling/benchmark_full_wallclock_runtime_gpu1.py` and
  `run_rld_training_population_scaling_v2.py`.
- Zebrafish LOFO8 preparation and complete model run:
  `../scripts/zebrafish/prepare_zebrafish_mprt_lofo8_seed42.py` and
  `run_zebrafish_mprt_lofo8_seed42.sh`; baseline evaluators and the aggregator
  are in the same directory.
- Zebrafish position-only StatAtlas/CRF-ID pairwise adaptations:
  `../scripts/zebrafish/evaluate_zebrafish_statatlas_crfid_lofo8.py`. The
  archived `select` locks, fold results, pair sufficient statistics and
  aggregate are under `zebrafish_lofo8/statatlas_crfid_pairwise_lofo8_v1/`.

Run `python scripts/check_release.py` from the repository root before a GitHub
push. It parses every Python and JSON file, checks the canonical fold-manifest
hash, enforces the main-table PASS/MISSING gates, and rejects accidentally
included dataset or checkpoint artifacts.

## Main benchmark status

`main_benchmark_seed42/VERIFIED_RESULTS.md` reproduces the populated Atanas and
Kato/RLD rows in the supplied main table. Its machine-readable evidence is in
`verified_fold_cells.csv`, `verified_summary.csv`, `artifact_audit.csv` and
`readiness.json`.

The following rows in the supplied document do not have five canonical PASS
cells and remain missing: `CRF_ID`, `GWOT-MD`, and `GWOT-MD (our adaptation)`
on both Atanas and Kato/RLD. Blank GWOT-MD cells are therefore correct. The
CRF-ID numbers shown in the supplied draft must not be presented as results of
the current canonical protocol without a new audited rerun.

## Single-specimen protocol

For each outer fold, the implementation computes centered/scaled geometry for
every outer-training animal, measures pairwise symmetric Chamfer distance, and
selects the training animal with minimum mean distance as the one fixed
template. Validation and test animals never participate in template selection.

The saved medoid audit passes, but the historical CV5 × 3-seed result combines
different fold families for some methods. It is retained because it appears in
the supplied experiment notes, but it is not a paired cross-method main table.
The canonical seed-42 table in `main_benchmark_seed42/` supersedes it.

## Result-version warnings

- Main claims use CV5 × seed42: Ours Top-1 is 74.92 ± 8.26% on Atanas and
  63.93 ± 4.61% on Kato/RLD. Values 74.24 ± 6.95% and 63.14 ± 4.45% are the
  historical CV5 × 3-seed stability estimates.
- `robustness/FINAL_ROBUSTNESS_TABLES.md` is retained for provenance but is
  explicitly historical and not benchmark-aligned. Use the corresponding
  `formal_native_cv5_*`, `formal_activity_noise_v1`, and
  `formal_dustbin_ablation_ours` directories for claims.
- The narrative values 61.10% (Ours) and 13.68% (fDNC) in the supplied
  robustness description refer to an older common-query analysis. They must
  not replace the formal native clean values 63.93% and 45.11%.
- The current mechanism summaries do not exactly equal every rounded number
  in the supplied draft. For example, the archived publication analysis gives
  Same similarities 0.8638/0.3603/0.6120 for geometry/activity/multimodal,
  while the draft says 0.8638/0.3633/0.6136. Use the CSV files in
  `mechanisms/` as the source of truth or explicitly label the older numbers.
- `../results/supplementary/model_development_cv5x3/` is a separate fold-pure
  Candidate A/B development study. It is not the source of the final NeuRID
  benchmark table.

## Third-party methods

Official third-party repositories are not vendored. Their URLs and frozen
commits are recorded in `../docs/THIRD_PARTY.md` and
`../baselines/official/manifests/official_repo_revisions.tsv`. Project-written
adapters and overlays are included. This keeps the GitHub package reviewable
without silently redistributing external repositories or weights.
