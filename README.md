# NeuRID / MPRT-Net

Code release for multimodal cross-animal neuron identity matching with
population-relational transport. The repository contains the primary MPRT-Net
implementation and the audited experiment entry points used for Atanas,
Kato/RLD, and zebrafish evaluation.

**Formal Atanas and Kato/RLD main-result protocol:** five locked biological
folds and the single model seed `42`. Reported means and sample standard
deviations are calculated over the five fold-level values. Deterministic
methods have no artificial seed. Any CV5 × 3-seed material in this repository
is historical provenance or a supplementary stability analysis, not a formal
main result. Zebrafish follows its separately specified LOFO8 × seed-42
protocol.

## Repository layout

- `mprt_net_v1_1/`: standalone primary model, training/evaluation code, and tests.
- `MODEL_CODE_GUIDE.md`: file-by-file guide to the complete NeuRID model implementation.
- `RUN_MODEL.md`: copy-and-run commands for one fold and formal CV5 × seed42.
- `RUN_DATASETS.md`: ready-to-copy commands for every included dataset.
- `RUN_ZM9624.md`: one-command two-worm leakage-controlled ZM9624 run.
- `results/`: compact protocols, audits, fold-level metrics, tables, and figures for the reported experiments.
- `scripts/benchmarks/`: clean CPD, fDNC, NuCLR, NGM-v2, FGW, and audit entry points.
- `scripts/benchmarks/crfid/`: CRF-ID preparation, audit, and aggregation tools.
- `scripts/fair_identity/`: train-set medoid/single-specimen reference protocol.
- `scripts/mprt/`: MPRT evaluation and component-ablation launchers.
- `scripts/mechanisms/`: cross-animal population-relation analyses and figures.
- `scripts/robustness/`: coordinate-noise, missing-neuron, and distractor experiments.
- `scripts/scaling/`: runtime and training-population scaling experiments.
- `scripts/zebrafish/`: zebrafish LOFO preparation, evaluation, and aggregation.
- `scripts/zm9624/`: ZM9624 preparation and two-direction held-out matching.
- `workflows/`: locked Atanas and Kato/RLD split preparation and manifests.
- `docs/protocols/`: evaluation and reporting contracts.
- `paper_submission_data/5fold_cross_validation/`: historical/supplementary
  CV5 × 3-seed provenance package; excluded from formal-result aggregation.

Raw datasets, checkpoints, large generated run directories, logs, caches,
local archives, and vendored third-party repositories are intentionally
excluded. Compact result records needed to trace the reported tables are
included under `results/`.

## Where is the complete model?

The complete NeuRID/MPRT-Net forward model is
[`mprt_net_v1_1/mprt_net/model.py`](mprt_net_v1_1/mprt_net/model.py). Its
geometry/activity encoders, population relations, relational transport,
Sinkhorn dustbin, losses, training and evaluation code are all in the same
package. See [`MODEL_CODE_GUIDE.md`](MODEL_CODE_GUIDE.md) for the module map.

## Install the primary model

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e './mprt_net_v1_1[eval,test]'
```

Run the unit tests with:

```bash
python -m pytest -q mprt_net_v1_1/tests
```

For an end-to-end train → train-only atlas → held-out test example, use
[`RUN_MODEL.md`](RUN_MODEL.md).

## Data contract

Each animal is stored as an NPZ file. The required arrays and validation rules
are documented in [`mprt_net_v1_1/DATA_CONTRACT.md`](mprt_net_v1_1/DATA_CONTRACT.md).
Dataset files are not included in this repository.

After preparing a dataset, run the numerical/data self-check before training:

```bash
python -m mprt_net.self_check --dataset-root /path/to/dataset --split train --device cpu
```

## Reproducing experiments

The canonical entry-point index is
[`docs/protocols/SCRIPT_INDEX.md`](docs/protocols/SCRIPT_INDEX.md). The unified
cross-validation and perturbation rules are in
[`docs/protocols/UNIFIED_BENCHMARK_PERTURBATION_PROTOCOL.md`](docs/protocols/UNIFIED_BENCHMARK_PERTURBATION_PROTOCOL.md).
The code-to-result map and protocol warnings are in
[`results/README.md`](results/README.md).

Examples:

```bash
python -m scripts.fair_identity.evaluate_train_reference_ensemble --help
python -m scripts.mechanisms.summarize_component_ablation_benchmark_seed42 --help
bash scripts/zebrafish/run_zebrafish_mprt_lofo8_seed42.sh all
python -m scripts.robustness.plot_rld_robustness_conditional_top1
```

Baseline repositories are not vendored in this release. Their URLs and frozen
revisions are recorded in [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md), with
project-specific overlays under `third_party_adapters/`.

## Release boundary

[`docs/GITHUB_SUBMISSION.md`](docs/GITHUB_SUBMISSION.md) records the exact
source-release scope and validation checks. Historical or unrelated research
artifacts remain outside the GitHub commit.

## License

The repository retains the upstream Apache-2.0 license. Individual third-party
methods remain subject to their own licenses.
