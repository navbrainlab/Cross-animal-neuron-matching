# Paper result-to-code manifest

Every quantitative result reported in *Learning Multimodal Population Relations
for Cross-Recording Neuron Identification* is represented below. Paths are
relative to the repository root. `results/` contains frozen reported outputs;
`scripts/` contains the corresponding evaluator, audit, or plotting code.

| Paper item | Frozen result | Reproduction/audit code |
|---|---|---|
| Worm Full main table | `results/main_benchmark_full/`, `results/main_benchmark_audit/` | `scripts/benchmarks/rescore_from_audit_package.py` |
| Worm Shared main tables | `results/main_benchmark_shared/` | `scripts/benchmarks/evaluate_uniform_medoid_covered_only.py`, `scripts/benchmarks/summarize_fair_identity_seed42.py` |
| Single-medoid reference | `results/single_specimen_medoid/` | `scripts/fair_identity/` |
| Per-animal Full/Shared boxplots | `results/per_animal/`, `results/figures/Atanas_Kato_boxplots.pdf` | `scripts/benchmarks/export_atanas_kato_per_animal.py`, `scripts/plot_paper_summaries.py` |
| Component and modality ablations | `results/component_ablation/seed42/` | `scripts/neurid/run_mprt_component_ablation_cv5x3.py`, `scripts/neurid/summarize_component_ablation_seed42.py` |
| Fixed-encoder matcher controls | `results/fixed_matchers_cv5_seed42/` | `scripts/neurid/evaluate_frozen_neurid_matchers_cv5.py` |
| Missing-atlas-relation control | `results/atlas_relation_mask_cv5_seed42/` | `scripts/neurid/run_atlas_relation_mask_ablation_cv5.py` |
| Activity margin and correction cases | `results/mechanisms/atanas_activity_margin_vs_ablation_gain/`, `results/mechanisms/atanas_activity_rescue_cases/` | `scripts/mechanisms/plot_activity_margin_vs_ablation_gain.py`, `scripts/mechanisms/plot_activity_rescue_case_heatmaps.py` |
| Relation specificity, margins, profiles, and population similarity | `results/mechanisms/` | `scripts/mechanisms/` |
| Qualitative activity predictions | `results/visualization_export_atanas_seed42/` | `scripts/neurid/export_atanas_visualization_csv.py` |
| Synthetic perturbation robustness | `results/robustness/formal_native_cv5_*`, `results/robustness/figures/` | `scripts/robustness/` |
| Unmatched-neuron rejection | `results/robustness/rejection_targets/`, `results/robustness/formal_rejection_comparison_ours/` | `scripts/robustness/eval_ours_rld_rejection_calibration_current_grouped.py`, `scripts/robustness/audit_ours_rld_rejection_targets.py` |
| Activity-window duration and position | `results/activity_window_cv5_seed42/`, `results/figures/activity_window_summary.pdf` | `scripts/neurid/evaluate_activity_window_stability_cv5.py`, `scripts/plot_paper_summaries.py` |
| Zebrafish LOFO8 table | `results/zebrafish_lofo8/` | `scripts/zebrafish/` and baseline adapters under `baselines/` |

The manuscript-renamed figure sources map as follows:

- `activity_margin_gain.pdf` → `atanas_activity_margin_vs_ablation_gain/activity_margin_vs_ablation_gain.pdf`
- `activity_correction_cases.pdf` → `atanas_activity_rescue_cases/activity_rescue_case_heatmaps.pdf`
- `relation_specificity.pdf` → `atanas_relation_profile_figure_f/relation_profile_similarity_figure_f.pdf`
- `robustness.pdf` → `robustness/figures/Figure_X_robustness_canonical_top1.pdf`
- `rejection_summary.pdf` → `robustness/rejection_targets/rejection_operating_curves.pdf`
- `relation_margin.png` → `mechanisms/atanas_margin_vs_top1/relation_margin_vs_mprt_top1_1x3.png`
- `overall_population_relation_similarity.png` → the file of the same name under `mechanisms/atanas_overall_similarity/`

The two summary layouts are regenerated under `results/figures/`.
`qualitative_activity.pdf` is a manuscript layout export; its complete
per-neuron coordinates, ground truth, and both model predictions are retained
in `results/visualization_export_atanas_seed42/` together with its exporter.

Historical candidate architectures, non-paper sensitivity studies, and their
outputs are intentionally excluded from this public tree.
