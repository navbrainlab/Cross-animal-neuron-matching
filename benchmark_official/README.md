# Official benchmark workspace

This directory is designed to live at:

`/home/ubuntu/klb/nuclr/nuclr/benchmark_official/`

## Principle

`third_party/` contains untouched official repositories.

`adapters/` contains all code written for this project to convert Atanas/RLD data to each official method's expected input and to convert its output into a common cross-animal matching score matrix.

Do **not** edit files under `third_party/`. If an official source requires a compatibility patch, put the patch under `patches/` and record it in provenance before applying it.

## Layout

```text
benchmark_official/
├── third_party/
│   ├── fdnc_official/
│   ├── wormid_official/
│   │   └── stat-atlas/          # pinned submodule used by WormID/NWBelegans
│   ├── neurpir_official/
│   ├── nuclr_official/
│   └── mult_official/
├── adapters/
│   ├── common/
│   ├── fdnc/
│   ├── stat_atlas/
│   ├── neurpir/
│   ├── nuclr/
│   ├── score_fusion/
│   ├── mult/
│   └── ours/
├── protocols/
│   ├── atanas/
│   └── rld/
├── envs/
├── manifests/
├── runs/
├── results/
├── logs/
└── scripts/
```

## Official repositories

- fDNC: `https://github.com/XinweiYu/fDNC_Neuron_ID`
- WormID/NWBelegans: `https://github.com/focolab/NWBelegans`
  - its `stat-atlas` git submodule pins the Statistical Atlas implementation
- NeurPIR: `https://github.com/ww20hust/NeurPIR`
- NuCLR: `https://github.com/nerdslab/NuCLR`
- MulT: `https://github.com/yaohungt/Multimodal-Transformer`
- GWTune: `https://github.com/oizumi-lab/GWTune`

GWTune is the authors' official generic single-matrix GWOT toolbox.  The
GWOT-MD neuronal-matching paper does not currently publish its multi-distance
implementation, so the latter must be reported as a paper-protocol
reimplementation; see `adapters/gwot_md/README.md`.

## Setup

Copy the supplied `setup_official_benchmark_repos.sh` to the NuCLR repository root and run:

```bash
cd /home/ubuntu/klb/nuclr/nuclr
conda activate nuclr310

bash setup_official_benchmark_repos.sh
python benchmark_official/scripts/audit_official_repos.py
```

The setup script records the exact git commit of every cloned repository in:

`benchmark_official/manifests/official_repo_revisions.tsv`

This commit manifest should be retained for the paper/rebuttal.

## Environment policy

Do not install all official dependencies into `nuclr310`.

- fDNC was developed/tested with Python 3.7.
- MulT targets Python 3.6/3.7 and older PyTorch.
- NuCLR targets Python 3.10.
- NeurPIR has its own requirements.

We will create method-specific environments after repository audit. Adapters should exchange NPZ/CSV/PT files so the benchmark remains reproducible across environments.

## Fair-comparison policy

All methods must use:

- the same outer test worms;
- the same eligible GT queries;
- validation-only model/hyperparameter selection;
- direct per-query Top-1 before Hungarian as the primary metric;
- identical Top-3, Top-5, MRR and Hungarian reporting;
- no test-set selection of score-fusion weights.

For methods whose original task is not pairwise correspondence (Statistical Atlas, NeurPIR, NuCLR, MulT), the paper must explicitly say `adapted using the authors' official implementation`.


## Important classification note: NeurPIR

The untouched official NeurPIR implementation is **not strictly activity-only**.
Its README states that the model consumes population neural activity together with
neuron coordinates; coordinates are used to choose/order nearest neighbours and to
create distance-bin positional embeddings.

Therefore the clean reporting options are:

1. `NeurPIR (official; activity + spatial context)` — preferred when claiming use of official code.
2. A spatially ablated NeurPIR — possible as a separate ablation, but it must be described as
   `NeurPIR adapted / spatially ablated`, not as the untouched official method.

The main benchmark should not silently label the untouched official model as activity-only.
