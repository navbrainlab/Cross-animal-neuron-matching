# Unified benchmark and perturbation protocol

## Status and scope

This is the reporting contract for the formal Atanas and Kato main benchmark
and for every experiment that is compared directly with it. It supersedes the
former five-fold × three-seed reporting convention for main-table claims.

## Locked protocol

| Analysis | Required data/model protocol | Required gate |
| --- | --- | --- |
| Main Benchmark | The locked biological CV5 split, model seed 42 | One value per fold; unweighted mean ± sample SD over the five folds |
| Component Ablation | The same CV5 membership and seed-42 Full checkpoint family | The Full arm must reproduce the main Benchmark fold by fold |
| Mechanism analyses containing Top-1 | Held-out predictions from the same CV5 × seed-42 runs | Query cohort, candidate universe, evaluator, and fold membership must match the Benchmark claim being explained |
| Robustness | The same CV5 × seed-42 checkpoints | `severity=0` must reproduce the native Benchmark result in every fold before any nonzero severity is reportable |
| Scaling compared with Benchmark accuracy | The same CV5 and seed 42 | Benchmark endpoint must reproduce fold by fold |
| Independent fixed-test scaling | A separately named and fully specified protocol | It must not be presented as a direct Benchmark-accuracy comparison |
| Deterministic methods (for example CPD) | No model seed; exactly the same five fold memberships | One deterministic value per fold; no artificial seed replication |

“Same CV5” means identical train, validation, and held-out test membership,
not merely five folds with similar sizes. Training and atlas construction use
outer-train only, checkpoint selection uses validation only, and the locked
held-out test set is read once for reporting.

For Top-1 mechanism analyses, “held-out predictions” means the saved
per-query predictions that generated the seed-42 Benchmark fold cells. A
mechanism package based on seeds 1/123, a different Full arm, refitted
predictions, a different test cohort, or a robustness common-query cohort is
historical/supplementary and cannot substantiate the main Benchmark result.

For Robustness, the severity-zero gate is method-native: checkpoint, fold,
query cohort, candidate universe, missing-candidate policy, evaluator, and
denominator must be unchanged from the clean Benchmark. A common-cohort
robustness estimand may be reported additionally, but it cannot replace this
native severity-zero reproduction gate.

## Audited values and current publication status

The following Ours values use the intended CV5 × seed-42 protocol:

| Dataset | Method | Top-1 |
| --- | --- | ---: |
| Atanas | Ours | **74.92 ± 8.26%** |
| Kato (`rld` internally) | Ours | **63.93 ± 4.61%** |

They are the locked Ours values. The new fail-closed aggregation is under
`runs/unified_main_benchmark_cv5_seed42_v1/`; its canonical manifest fixes exact fold membership,
and `artifact_audit.csv` admits a method/dataset group only when all five cells pass. Every
populated method row in `results/main_benchmark_seed42/VERIFIED_RESULTS.md` now has five PASS
cells on both datasets. In particular, Kato fDNC, NuCLR, and GeoTransformer have audited Top-1
values of 45.11 ± 16.99%, 10.14 ± 7.62%, and 16.52 ± 13.33%, respectively. These supersede the
old-fold seed-42 values 26.41%, 15.17%, and 25.03%.

The only unresolved main-table groups are CRF-ID, GWOT-MD, and GWOT-MD (our adaptation), all
marked five-fold `MISSING` on both datasets in `results/main_benchmark_seed42/readiness.json`.
No draft or historical mixed-fold value may fill those cells without a new audited run.

The former CV5 × three-seed values, including **74.24 ± 6.95%** and
**63.14 ± 4.45%**, are not main-table results. They may be retained only as
historical provenance or as an explicitly labelled supplementary stability
check; they must never be substituted for the values above.

## Publication gates

An output may be labelled benchmark-aligned only if its machine-readable
protocol records the fold manifest, model seed (or deterministic status),
checkpoint hashes where applicable, evaluator and cohort definition,
fold-level metrics, and a passed fold-by-fold clean reproduction gate.
Failing or missing gates are fail-closed: retain the artifact, label it
historical or pending, and do not use its table or figure for a main claim.
