# Canonical main-table audit package — scoring audit revision

No model was trained, tuned, or selected while building this package. Most files are copied or decoded from saved artifacts. Because the original NGM-v2 evaluator did not persist score matrices, its frozen seed-42 checkpoints were replayed once to export scores; every replayed prediction row is guarded against the original saved `query_level.csv`.

## Publication protocol represented here

- Five biological outer folds; learned-model seed 42.
- Evaluation split: held-out outer test animals.
- Denominator: every canonical test query with a valid unique ground-truth identity that occurs in the strict outer-training identity union.
- Every method is evaluated on this same canonical set.
- For a single-medoid method, a canonical GT identity absent from the frozen medoid is an automatic error.
- Fold metrics are computed first; the reported result is the unweighted five-fold mean and sample standard deviation.
- Checkpoint/model selection is validation-only where selection is applicable.

## Start here

1. `main_table/UPDATED_MAIN_TABLE.md`: current table and denominator audit.
2. `main_table/canonical_query_manifest.csv`: exact canonical query keys.
3. `predictions/canonical_complete_predictions.csv`: one complete canonical view per available method, with absent-medoid rows explicitly added as errors.
4. `predictions/native/`: unmodified existing prediction CSVs.
5. `../../scripts/benchmarks/rescore_from_audit_package.py`: portable rescorer that reproduces the table from this package alone.
6. `manifests/fold_split_manifest.csv`: five-fold train/validation/test assignment plus hashes.
7. `references/method_reference_summary.csv`: reference UID/type and candidate-list routing.
8. `provenance/checkpoint_sources.csv`: seed, checkpoint, hashes, and selection metadata.
9. `KNOWN_LIMITATIONS.md`: fields that were not saved and are therefore not invented.

## Added scoring-audit material

- `predictions/ngmv2_score_export/`: complete NGM-v2 score matrices, candidate column order, stable Top-5 decoding, and tie diagnostics. Every fold reports `EXACT_SAVED_PREDICTION_REPLAY`.
- `predictions/ngmv2_top5_tie_summary.csv`: competition-rank versus fractional-tie sensitivity on both canonical denominators.
- `../../scripts/fair_identity/`, `../../neurid/mprt_net/`, and `../../scripts/lib/`: the maintained NeuRID query exporter, loader, matcher, ranking, and shared benchmark code.
- The current-CV NuCLR wrapper and `references/nuclr_integer_identity_mapping.csv` preserve the exact NuCLR evaluation and integer decoding path.
- `predictions/crfid_raw_matlab/`: original CRF MATLAB outputs and sidecars.
- `manifests/raw_neuron_index_manifest.csv`: raw cell labels, all validity masks, original row, post-finite row, and strict canonical eligibility without activity matrices.

Dataset NPZs are available under `../../data/`; checkpoint binaries are not duplicated. See `PUBLIC_RELEASE_NOTE.md` for the public-package boundary.
