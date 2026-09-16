# NeuRID

Official code and prepared evaluation data for **Learning Multimodal
Population Relations for Cross-Recording Neuron Identification**.

This repository intentionally contains one NeuRID implementation: the model
described in the manuscript. Historical MPRT-Net implementations, dynamic
atlas variants, transfer variants, candidate-training variants, and their
result folders are not part of this public tree.

## Contents

- `neurid/mprt_net/`: the complete manuscript model, loss, training, and data loader.
- `data/`: prepared Atanas, Kato/RLD, and zebrafish folds used by the study.
- `scripts/neurid/run_cv5.py`: the locked seed-42 five-fold worm runner.
- `results/`: every manuscript result, including fold/query-level audit data.
- `baselines/`: comparison-method adapters and upstream-code notes.
- `workflows/`: source-data extraction and split-construction provenance.
- `docs/`: architecture, data provenance, and release-boundary documentation.

No trained checkpoints, caches, or alternative NeuRID versions are included.
Frozen numeric outputs used in the paper are included under `results/`.

## Install

Python 3.10 or newer is required. Confirm with `python --version` before
installing.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e './neurid[test]'
```

## Verify the release

```bash
python scripts/check_release.py
python -m pytest -q neurid/tests
sha256sum -c data/SHA256SUMS
```

## Run the worm experiments

Run a quick smoke experiment on one fold:

```bash
python scripts/neurid/run_cv5.py --datasets atanas --folds 0 --epochs 1 \
  --pairs-per-epoch 1 --device cpu
```

Run the formal five-fold, seed-42 protocol:

```bash
python scripts/neurid/run_cv5.py --datasets atanas,kato_rld --device cuda
```

Outputs are written under `runs/` and are ignored by Git. See
[`RUN_DATASETS.md`](RUN_DATASETS.md) for dataset paths and the zebrafish layout.
For paper provenance, see
[`results/PAPER_RESULTS_MANIFEST.md`](results/PAPER_RESULTS_MANIFEST.md).

## Primary model entry points

- Architecture: `neurid/mprt_net/model.py`
- Training and fixed-atlas evaluation: `neurid/mprt_net/train.py`
- Input validation/loading: `neurid/mprt_net/data.py`
- Relation-conditioned attention: `neurid/mprt_net/layers.py`
- Matching loss: `neurid/mprt_net/losses.py`
- Log-domain Sinkhorn and relation cost: `neurid/mprt_net/sinkhorn.py`

The Python API exposes `NeuRID`/`MPRTNet`, `ModelConfig`, and `MPRTOutput`.

## Data and licensing

The repository includes processed NPZ files and locked split metadata. Their
scientific sources and preparation routes are documented in
[`data/README.md`](data/README.md). Before making the repository public, the
maintainer must confirm that redistribution of each processed dataset is
consistent with its upstream terms. Code is covered by `LICENSE`; upstream
datasets and baseline implementations retain their own terms.
