# Zebrafish LOFO8 test supplement

This directory contains the complete eight-fold test record for all 12 methods
in the zebrafish table. The statistical unit is the held-out fish: learned
methods have one run with seed 42, deterministic methods have no seed, and the
reported uncertainty is the sample SD (`ddof=1`) of the eight fold values.

## Files

- `ABLATION_RESULTS_ZH.md`: Chinese summary of the four NeuRID ablations
  (`full`, `no_transport`, `geometry_only`, and `activity_only`), including
  paired Top-1 effects, confidence intervals, and limitations. The exact
  train/evaluate launcher is
  `../../scripts/zebrafish/run_zebrafish_mprt_lofo8_seed42.sh`; its aggregation
  code is `../../scripts/zebrafish/aggregate_zebrafish_mprt_lofo8_seed42.py`.
- `all_run_metrics.csv` and `zebrafish_mprt_lofo8_seed42_summary.{json,md}`:
  machine-readable fold-level ablation results and their held-out-fish
  aggregate.
- `reported_fold_results.csv`: the authoritative 96 method × fold table. The
  StatAtlas and CRF_ID† rows come from the new Table-1-core locked reruns; all
  other rows retain their prior locked-test records.
- `test_pairs.csv`: all 101 physical test pairs, source-file hashes, query
  denominators, and the exact ordered q/r candidate lists.
- `table1_core_reference_manifest.csv`: the 101-pair shared leave-two-out
  reference-timepoint manifest used by the updated StatAtlas and CRF_ID† rows;
  it is also the reference information made available to retained pairwise
  baselines, whose unchanged cores do not consume additional atlas records.
- `query_predictions/<method>/fold_<k>.csv.gz`: 205,872 rows in total, one row
  per method and directed query. Every row contains the target, predicted
  candidate, rank, Top-1/Top-5/RR terms, Hungarian prediction, seed, pair and
  candidate-order hash.
- `query_evaluation/paired_top1_ours_vs_crfid.{csv,json}`: the eight
  held-out-fish Top-1 differences and their two-sided paired t confidence
  interval.
- `REFERENCE_CONFIGS.json`: seeds, pinned upstream revisions, exact reference
  configurations, validation candidate grids and their declared order, and
  the selected checkpoint/configuration for each fold.
- `BASELINE_CORE_AUDIT.md`: Table 1 versus former Table 2 implementation audit
  and the keep/rerun decision for every row.
- `query_evaluation/`: fold/pair metrics rebuilt only from the query rows,
  reported-result aggregation, and a reported-versus-replay comparison.
- `ZEBRAFISH_BENCHMARK_PROTOCOL.json`: the compact benchmark contract including
  the Table-1-core baseline correction.

The main result table rebuilt from the original fold records is
`query_evaluation/reported_summary.md`. The independently rebuilt table from
the archived query rows is `query_evaluation/summary.md`.

Across the same eight held-out fish, NeuRID improves Top-1 over CRF_ID by
1.05 pp (paired 95% CI [-1.00, 3.09] pp; two-sided paired t interval). Because
the interval includes zero, the eight-fish result does not establish a stable
positive improvement at the 95% confidence level.

## Table 1 baseline consistency update

The reported Table 2 rows now use the same core implementations as Table 1.
For every original physical q/r test pair, both endpoint timepoints are excluded
and all remaining unique timepoints of that fish form the common reference
atlas. Stable `original_row` tracking IDs align reference observations. This is
a multi-reference input-condition change; any comparison must give every method
the same reference information.

`StatAtlas` calls the exact Table 1 `train_official_atlas`, `label_free_align`
and `atlas_identity_posterior` functions. Its pair score is the same posterior
overlap `P_q @ P_r.T`, followed by the shared Hungarian implementation. The new
eight-fold result is 89.57 +/- 17.84% Top-1. Its complete compact archive is
`statatlas_table1_core_lofo8_v1/`.

The audit found that the former `CRF-ID (pairwise Python adaptation)` was not the
Table 1 algorithm. `CRF_ID†` was therefore rerun with the byte-identical Table 1
MATLAB adapter: fully connected angle CRF, uniform node potential, official UGM
LBP and iterative duplicate resolution. The q/r score is belief overlap
`B_q @ B_r.T`, followed by the shared Hungarian implementation. Its new
eight-fold result is 96.64 +/- 2.05% Top-1. The archive is
`crfid_table1_core_lofo8_v1/`.

Both archives contain all eight validation locks, test pair manifests,
per-query predictions, fold metrics and replay validation. Both methods are
deterministic (`seed=null`) and geometry-only (`GG`). The former single-reference
pairwise adaptations remain under `statatlas_crfid_pairwise_lofo8_v1/` solely as
historical provenance and are no longer Table 2 rows. The separate all-other-
timepoint StatAtlas experiment remains under `statatlas_multireference_lofo8_v1/`.
The identical shared manifest used by both updated methods is published as
`table1_core_reference_manifest.csv`. Other baseline results are retained only
where their core and q/r evaluation code are unchanged; those pairwise methods
receive this manifest as protocol metadata but, by definition, ignore extra
atlas-reference records.

## Vanilla FGW modality audit

Vanilla FGW is labelled **GG**. Its feature cost is squared Euclidean distance
between normalized XYZ coordinates, and both structural costs are
within-population Euclidean distances computed from XYZ. No activity array is
loaded or passed to the matching function; `activity_used=false` and
`activity_weight=0.0`. POT's `alpha` mixes these two geometry-derived terms and
must not be interpreted as an activity weight. The executable source is
`../../scripts/zebrafish/evaluate_zebrafish_vanilla_fgw_lofo8.py`, and the
machine-readable declaration is `methods.vanilla_fgw.config` in
`REFERENCE_CONFIGS.json`.

## Official FUGW G+A supplement

FUGW uses the official dense `fugw==0.1.1` implementation. Train-only
per-axis-z-scored XYZ supplies its linear feature term; normalized within-side
RMS distances between the archived 128-point activity traces supply its GW
structural term. Thus the official alpha has the explicit semantics
`alpha * activity_GW + (1-alpha) * geometry_Wasserstein`.

Each fold selects `(alpha, rho, eps)` on the outer validation fish only from
the declared `3 x 3 x 1` grid, durably writes `LOCKED_BEFORE_TEST.json`, and
only then opens the test directory. The eight selected triples and solver
settings are recorded in `REFERENCE_CONFIGS.json`. The executable runner and
coupling-replay auditor are `../../scripts/zebrafish/evaluate_zebrafish_fugw_ga_fold.py`
and `../../scripts/zebrafish/audit_zebrafish_fugw_ga_lofo8.py`.

## Candidate and reference convention

For a physical pair, q and r are the filenames recorded in `test_pairs.csv`.
The two evaluated query directions are `q_to_r` and `r_to_q`; the opposite side
is the reference/candidate population. Physical pairs are sorted by `pair_id`.
Candidates retain their original NPZ row order after only the finite-XYZ and
`valid_xyz_mask` filter. Labels never filter the candidate population.

`candidate_order_sha256` binds each prediction to its ordered reference list.
The corresponding lists and hashes are in `test_pairs.csv`. GeoTransformer
internally replaces supervised identities by stable integer IDs, so its hash
binds the same ordered nodes in that internal representation; candidate index
is the cross-method canonical key.

Ranks use `1 + count(score > target_score)`. Thus score ties receive the
optimistic benchmark rank. The archived predicted index resolves an exact
maximum tie by the first candidate in the declared order. GeoTransformer
retains its native exposed-candidate Top-1/Top-5 rules. Hungarian predictions
use one physical q/r assignment except fDNC, whose native implementation makes
independent directional assignments.

## Validate and aggregate

From the repository root:

```bash
PYTHONPATH=neurid:. python scripts/zebrafish/aggregate_zebrafish_query_records.py \
  --prediction-root results/zebrafish_lofo8/query_predictions \
  --test-pairs results/zebrafish_lofo8/test_pairs.csv \
  --reported-fold-results results/zebrafish_lofo8/reported_fold_results.csv \
  --output-root results/zebrafish_lofo8/query_evaluation
```

The command fails on a missing fold, wrong seed, duplicate/missing query,
unknown pair, candidate-count mismatch, or a replay delta larger than 0.003.
The checked archive contains the ten manuscript methods, 80 fold records,
1,010 method-pair records and 171,560 method-query records
(`VALIDATION.json`: `PASS`).

The prediction archive was replayed on CPU from the same locked checkpoints.
The original reported tests used CUDA for learned third-party methods. Near
ties can therefore move a few individual predictions. The largest absolute
fold-level reported/CPU-replay difference is recorded in
`query_evaluation/VALIDATION.json` (at most 0.002605 in this archive); no value
is silently substituted. Use `reported_summary.md` for the paper numbers and
`summary.md` when auditing the included row-level predictions.

## Re-exporting predictions

The model-specific replay entry points are:

- CPD: `evaluate_zebrafish_cpd_lofo8.py --query-output-dir ...`
- NeuRID: `python -m mprt_net.evaluate --query-output ... --fold ...`
- NuCLR: `evaluate_zebrafish_nuclr_locked.py --query-output ... --fold ...`
- fDNC, Vanilla FGW and NGM-v2: the corresponding
  `export_zebrafish_*_queries.py` scripts.
- Official FUGW G+A: `evaluate_zebrafish_fugw_ga_fold.py`; its independent
  replay is `audit_zebrafish_fugw_ga_lofo8.py`.
- StatAtlas: `evaluate_zebrafish_statatlas_table1_lofo8.py`.
- CRF_ID†: `evaluate_zebrafish_crfid_table1_lofo8.py` plus the MATLAB files in
  `scripts/zebrafish/matlab/`.
- GeoTransformer: the release overlay's `evaluate_zebrafish_locked.py` accepts
  `--query-output`, `--fold`, and CPU-compatible `--num-workers 0`.

Third-party checkouts and checkpoints are not included. Clone the pinned
revisions in `docs/THIRD_PARTY.md`; supply checkpoint/data paths as shown by
each script's `--help`. To regenerate `test_pairs.csv`, run:

```bash
PYTHONPATH=neurid:. python scripts/zebrafish/build_zebrafish_test_pair_records.py \
  --data-root /path/to/Zebrafish_MPRT_LOFO8_60m \
  --output results/zebrafish_lofo8/test_pairs.csv
```
