# Active script index

Executable research tools are grouped under `scripts/`; the repository root contains only standard project metadata. Run Python entry points from the repository root with `python -m scripts.<group>.<module>` when possible.

| Directory | Scope |
|---|---|
| `scripts/lib/` | Shared locked-CV and fair-identity protocol helpers |
| `scripts/fair_identity/` | Train-medoid benchmark evaluation and summaries |
| `scripts/neurid/` | Primary NeuRID training, evaluation, and ablations |
| `scripts/benchmarks/` | Official CPD/fDNC/NuCLR/NGM-v2/FGW benchmark runners |
| `scripts/robustness/` | Shared-manifest RLD corruption experiments and figure generation |
| `scripts/scaling/` | Full-pipeline GPU runtime and training-population scaling analyses |
| `scripts/zebrafish/` | Zebrafish LOFO8 preparation, training, evaluation, and aggregation |
| `scripts/mechanisms/` | Mechanism analyses and figure generation |
| `scripts/zm9624/` | Two-worm leakage-controlled preparation, training, and held-out matching |

Primary commands:

```bash
python -m scripts.neurid.run_model --help
bash scripts/zm9624/run_strict_loso.sh
python -m scripts.fair_identity.evaluate_train_reference_ensemble --help
python -m scripts.mechanisms.summarize_component_ablation_benchmark_seed42 --help
python -m scripts.neurid.summarize_cv5_seed42 --help
bash scripts/zebrafish/run_zebrafish_mprt_lofo8_seed42.sh all
python scripts/zebrafish/evaluate_zebrafish_statatlas_crfid_lofo8.py --help
python -m scripts.robustness.plot_rld_robustness_conditional_top1
python -m scripts.scaling.benchmark_full_wallclock_runtime_gpu1 --help
python -m scripts.scaling.run_rld_training_population_scaling_v2 --help
```

Scripts whose names contain `cv5x3` are retained for provenance or supplementary stability
checks; they are not launchers for the formal main protocol. Formal learned-method runs use the
five locked folds with `seed=42` only.

The zebrafish StatAtlas/CRF-ID entry point is self-contained in
`scripts/zebrafish/evaluate_zebrafish_statatlas_crfid_lofo8.py`. Its `select`,
`test`, and `aggregate` stages enforce validation-first parameter locking.
Because zebrafish IDs are local to longitudinal pairs, these are explicitly
pairwise adaptations rather than canonical global named-identity atlas runs.

Historical smoke/pilot entry points are stored under `archive/workspace_cleanup_20260827/root_entrypoints/`. Non-primary model implementations remain under `archive/non_primary_models_20260825/`.

Cross-domain, Chaudhary, unrelated-dataset workflows, exploratory diagnostics,
and superseded root launchers remain outside this release. Scaling code is now
included because runtime and training-population scaling are part of the
reported experiment set.

Mechanism tables intended to support the main clean Benchmark must follow
`docs/protocols/MECHANISM_BENCHMARK_PROTOCOL.md`; in particular, use seed 42 and five biological
folds rather than pooling the historical 5-fold × 3-seed cells.
