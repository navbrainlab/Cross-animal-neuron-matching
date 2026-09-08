# GWOT-MD / Atanas-21：论文 Top-5 严格复现包

目标是复现论文 *Unsupervised Neuronal Matching with Spontaneous Neuronal
Activity* 在自由运动 Atanas 数据上的 Appendix-F 指标：9 个 teacher，内部
leave-one-out 选择 `h`，`v=5, k=5` majority-vote Top-5，论文报告中位数 46%。

## 和先前 23.08% 结果的区别

先前结果不是论文复现：它使用 38 worms/CV5、非论文的活动距离和另一种
GW 求解器。本包使用独立缓存，绝不读取该结果。

论文队列是官方 manifest 中恰好 21 条 `label=true` 且类型同时包含
`baseline, neuropal` 的记录，并不是最早 21 条 NeuroPAL，也不包括 heat
记录。其 Figure C.1 对角线为：

```text
78,87,95,74,84,69,91,95,67,111,91,90,71,85,60,92,87,87,96,95,82
```

严格评估入口会检查完整 21×21（441 个）共同标签计数；任意一个不一致就
拒绝输出“论文复现”结论。

## 方法锁定

- 所有记录神经元均参加无监督匹配；不读取 xyz，不做 geometry cohort 筛选。
- `F/F20`（对 delayed cosine 而言逐神经元正比例缩放不改变结果），一阶
  0.01 Hz Butterworth 高通，零相位滤波。
- `d_tau = 1 - cosine(u[0:T-tau], v[tau:T])`，每个整数 lag，`w_tau=1`。
- `h = 0,5,...,50`。
- uniform neuron masses。
- 21 个 `epsilon = 10^-4, 10^-3.8, ..., 1`。
- 每个 epsilon 50 个随机可行初值；共 1,050 个解，按未正则化 GWOT-MD
  目标选择最优解。
- Appendix F：1,000 个不重复的 9-teacher 组合；8-teacher 内部 LOO 选 h；
  剩余 12 worms 用 9 teachers 评估；`noID` 占 Top-v 槽位但不进入最终 Top-k。

## 安装与自检

```bash
python -m pip install -r requirements.txt
python solve_gwot_md_atanas21.py self-check
```

## 数据预检

支持两种输入：

1. 每个 recording 一个 NPZ，包含 `activity_raw`（或 README 中脚本列出的别名）、
   `cell_id` 和 `recording_uid`。NPZ 已含标签时不需要 `--labels`。
2. Atanas 官方 `processed_h5.tar.bz2` 解压后的 H5，再加标签 JSON/JSON.bz2。

```bash
python solve_gwot_md_atanas21.py validate-data \
  --data-dir /data/atanas_h5 \
  --labels /data/neuropal_label.json.bz2 \
  --verify-h5-sha256
```

注意：WormWideWeb/Zenodo v4（2026-04）汇总标签单独使用时并不逐格复现
论文 Figure C.1。仓库现已提供
`benchmark_official/adapters/gwot_md/prepare_atanas21_paper_inputs.py`：它组合
20 条本地旧 DANDI clean-mask 标签、v4 中缺失记录 `2023-01-19-01` 的标签，
以及 v4 新补的 `2023-01-23-21/AVJL`，可重建与 Figure C.1 全部 441 格一致
的评分标签。该组合是“论文分母等价重建”，不是作者发布的历史标签快照；
脚本会保存逐记录来源并在任何矩阵差异时默认拒绝生成输入。

## 计算 pair caches

论文设置包含 11×21×20 = 4,620 个有向 pair/h 任务、总计 4,851,000 个
epsilon/initialization starts，适合 Slurm/任务数组，而不适合单机交互运行。
下面示例分成 64 个可恢复 shard；把 `SLURM_ARRAY_TASK_ID` 换成本机的
`0..63` 也可以。

```bash
python solve_gwot_md_atanas21.py solve \
  --data-dir /data/atanas_h5 \
  --labels /data/paper_label_snapshot.json.bz2 \
  --verify-h5-sha256 \
  --device cuda \
  --init-batch-size 50 \
  --num-shards 64 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --output-template '/runs/atanas21_gwot_md_paper_h{h}'
```

`--device cuda` 需要按机器 CUDA 版本另行安装 PyTorch。50 个初值在一个
GPU batch 中并行；显存不足时把 `--init-batch-size` 改成 10 或 25，不改变
初值集合。不要改 solver tolerance/max-iter；一旦修改，`config.json` 的 `paper_mode`
会变为 false。任务已存在时默认 resume。

完成后检查 4,620 个缓存：

```bash
python solve_gwot_md_atanas21.py audit \
  --output-template '/runs/atanas21_gwot_md_paper_h{h}'
```

## 论文 majority-vote Top-5

将 `paper_exact_h_selection_majority_vote.py` 放在本脚本同目录（发布压缩包
已包含），执行：

```bash
python evaluate_paper_top5_strict.py \
  --run-template '/runs/atanas21_gwot_md_paper_h{h}' \
  --h-values 0,5,10,15,20,25,30,35,40,45,50 \
  --teacher-count 9 \
  --num-splits 1000 \
  --selection-v 5 \
  --selection-k 5 \
  --v-values 5 \
  --k-values 5 \
  --seed 42 \
  --tie-break first \
  --output-dir /runs/atanas21_paper_top5
```

最终比较字段是 `aggregate` 中 `v=5, k=5` 的
`paper_median_individual_accuracy`，不是 direct pairwise Top-5、coverage 条件
准确率或 pooled micro accuracy。

## “严格”的边界

论文公开文本没有给出：随机种子、50 个初值的精确采样、Sinkhorn/外循环
停止条件、Butterworth 是单向还是零相位、投票相同票数的裁决规则，也未公开
GWOT-MD 代码或匹配缓存。因此，在拿到作者代码/缓存前，任何程序都不能诚实
保证 bit-for-bit 得到 46%。本包做到的是：

- 对所有论文已披露的条件做硬校验；
- 对未披露项采用固定、可审计的实现并写入 `config.json`；
- 绝不通过调 seed、标签分母或 coverage 把结果“调到 46%”。

作者团队公开的 GWTune 是单距离矩阵 GWOT 工具箱，不包含 GWOT-MD 的多延迟
矩阵接口；核验记录见 `OFFICIAL_CODE_AUDIT.json`。因此本目录结果应写作
“GWOT-MD 论文协议重实现”，不能写作“官方 GWOT-MD 代码结果”。若 Figure
C.1 检查或 cache 检查失败，结果只能标为近似复现，不能标为严格复现。

可运行下面的边界审计。它在真实 Atanas pair 上比较官方 GWTune 与本重实现
退化到单距离矩阵的 `h=0` 情形；当前结果的初值逐元素相同，传输矩阵最大绝对
误差为 `2.93e-18`，目标函数绝对误差为 `1.11e-16`：

```bash
benchmark_official/adapters/gwot_md/run_atanas21_paper_reproduction.sh official-h0-audit
```

结果保存在 `GWTUNE_H0_EQUIVALENCE.json`。该检查也锁定了 POT/GWTune 的 2 倍
梯度约定，防止把论文 epsilon 网格无意中整体平移一个 2 倍尺度。

## 来源

- 论文：https://openreview.net/forum?id=qAgQqVVwq9
- Atanas et al. 数据论文：https://doi.org/10.1016/j.cell.2023.07.035
- 官方数据说明：https://wormwideweb.org/about/datasets/
- 官方 manifest：https://github.com/flavell-lab/WormWideWeb-data/blob/main/activity/raw/atanas_kim_2023.csv
- Zenodo 数据：https://zenodo.org/records/19388374
- Oizumi-lab GWTune 初始化/求解惯例：https://github.com/oizumi-lab/GWTune
