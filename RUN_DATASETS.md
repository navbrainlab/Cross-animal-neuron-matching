# 各数据集直接运行命令

## 首次安装

```bash
cd /home/ubuntu/klb/nuclr/nuclr/NeuRID_reproducibility
/home/ubuntu/anaconda3/envs/nuclr310/bin/python -m pip install -e './neurid[eval,test]'
```

## Atanas（正式 5-fold、seed=42）

```bash
cd /home/ubuntu/klb/nuclr/nuclr/NeuRID_reproducibility
/home/ubuntu/anaconda3/envs/nuclr310/bin/python -m scripts.neurid.run_model \
  --dataset-root /home/ubuntu/klb/nuclr/nuclr/Data/Atanas_SF_unified_000776/cv5_grouped_v1 \
  --output-root /home/ubuntu/klb/nuclr/nuclr/runs/atanas_github_release_cv5_seed42_v1 \
  --dataset-name atanas \
  --device cpu
```

结果：`/home/ubuntu/klb/nuclr/nuclr/runs/atanas_github_release_cv5_seed42_v1/SUMMARY.md`

## Kato/RLD（正式 5-fold、seed=42）

```bash
cd /home/ubuntu/klb/nuclr/nuclr/NeuRID_reproducibility
/home/ubuntu/anaconda3/envs/nuclr310/bin/python -m scripts.neurid.run_model \
  --dataset-root /home/ubuntu/klb/nuclr/nuclr/Data/Dunn_001623/cv5_grouped_v1 \
  --output-root /home/ubuntu/klb/nuclr/nuclr/runs/rld_github_release_cv5_seed42_v1 \
  --dataset-name rld \
  --device cpu
```

结果：`/home/ubuntu/klb/nuclr/nuclr/runs/rld_github_release_cv5_seed42_v1/SUMMARY.md`

## Zebrafish（LOFO8、seed=42）

```bash
cd /home/ubuntu/klb/nuclr/nuclr/NeuRID_reproducibility
REPO_ROOT=/home/ubuntu/klb/nuclr/nuclr/NeuRID_reproducibility \
PYTHON_BIN=/home/ubuntu/anaconda3/envs/nuclr310/bin/python \
MPRT_ROOT=/home/ubuntu/klb/nuclr/nuclr/NeuRID_reproducibility/neurid \
SOURCE_ROOT=/home/ubuntu/klb/nuclr/nuclr/Data/Zebrafish_LOFO8_joint_from_scratch \
DATA_ROOT=/home/ubuntu/klb/nuclr/nuclr/Data/Zebrafish_MPRT_LOFO8_60m_github_v1 \
RUN_ROOT=/home/ubuntu/klb/nuclr/nuclr/runs/zebrafish_github_release_lofo8_seed42_v1 \
LEGACY_FOLD1_FULL=/home/ubuntu/klb/nuclr/nuclr/runs/zebrafish_github_release_lofo8_seed42_v1/fold_1/seed42/full \
DEVICE=cpu \
GPUS=0 \
bash scripts/zebrafish/run_zebrafish_mprt_lofo8_seed42.sh all
```

结果：`/home/ubuntu/klb/nuclr/nuclr/runs/zebrafish_github_release_lofo8_seed42_v1/aggregate/`

## ZM9624（两只虫双向留一、seed=42）

```bash
cd /home/ubuntu/klb/nuclr/nuclr/NeuRID_reproducibility
ZM_PYTHON_BIN=/home/ubuntu/anaconda3/envs/nuclr310/bin/python \
ZM_DATA_ROOT=/home/ubuntu/klb/zm9624 \
ZM_OUTPUT_ROOT=/home/ubuntu/klb/nuclr/nuclr/runs/zm9624_github_release_seed42_v1 \
ZM_DEVICE=cpu \
bash scripts/zm9624/run_strict_loso.sh
```

结果：`/home/ubuntu/klb/nuclr/nuclr/runs/zm9624_github_release_seed42_v1/`

当前机器不能连接 NVIDIA 驱动，所以命令统一使用 CPU。在有可用 NVIDIA
GPU 的机器上，可将 `cpu` 改成 `cuda`。
