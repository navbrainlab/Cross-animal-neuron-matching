# Fold-pure Candidate A/B：5-fold CV × 3 seeds

Candidate A：HyQuRP + Compact NuCLR + concat + Population Transformer（无 Stage 2、无 Sinkhorn）。
Candidate B：HyQuRP + Compact NuCLR → Stage 2 → Population Transformer（有 Stage 2、无 Sinkhorn）。

主估计先在每个 outer fold 内平均 3 seeds，再对 5 个 fold means 计算非加权 mean ± sample SD。
B−A 的 95% CI 以 outer fold 为 cluster，进行 10,000 次 percentile bootstrap；仅 5 folds，CI 为描述性。

## ATANAS

| Candidate | Ranking Top-1 | MRR | Assignment Top-1 |
|---|---:|---:|---:|
| A | 54.41% ± 8.42% | 64.90% ± 8.36% | 53.74% ± 8.36% |
| B | 59.35% ± 8.52% | 69.07% ± 8.18% | 58.61% ± 8.68% |

| B − A | Effect | 95% fold-bootstrap CI |
|---|---:|---:|
| Ranking Top-1 | +4.94 pp | [+3.01, +6.35] pp |
| MRR | +4.18 pp | [+2.63, +5.52] pp |
| Assignment Top-1 | +4.86 pp | [+3.42, +5.95] pp |

## RLD

| Candidate | Ranking Top-1 | MRR | Assignment Top-1 |
|---|---:|---:|---:|
| A | 49.12% ± 9.22% | 63.02% ± 8.57% | 44.34% ± 8.63% |
| B | 52.98% ± 8.54% | 66.03% ± 7.52% | 48.32% ± 9.43% |

| B − A | Effect | 95% fold-bootstrap CI |
|---|---:|---:|
| Ranking Top-1 | +3.87 pp | [+2.39, +5.41] pp |
| MRR | +3.01 pp | [+1.51, +4.51] pp |
| Assignment Top-1 | +3.98 pp | [+2.52, +5.65] pp |

## 结果判定

本次 fold-pure CV 支持 **Candidate B（Stage 2、无 Sinkhorn）** 作为两者中的优选模型。
Atanas 的 Ranking Top-1 提升为 +4.94 pp，5/5 folds 均为正；
RLD 的提升为 +3.87 pp，5/5 folds 也均为正。
三个 seeds 各自的跨折平均 B−A 效应在两个数据集上均为正。
因此保留 Stage 2、移除 Sinkhorn 是当前两候选中由严格 fold-pure 证据支持的选择。

## Fold-pure 审计

- 每个 dataset/fold/seed 都使用独立的 NuCLR 权重链路；30 个 encoder audit 全部通过。
- fold-local source NuCLR 仅使用 outer-train、训练 30 epochs、test dataloader 引用数为 0、未打开 identity labels。
- source T2/ST2 checkpoint 经权重保留截断为 T1/ST1，再只用同一 outer-train fine-tune；inner validation 选择 Compact NuCLR checkpoint。
- NuCLR checkpoint 锁定后才导出 sealed outer-test activity representation。
- A/B matcher 只用 outer-train 训练、inner validation 选 checkpoint，选定后才读取 outer-test list。
- Atanas 为 38 worms；RLD 为 evaluable93，预先排除两条 clean identity 数为 0 的 recordings。
- RLD 和 Atanas 都按 acquisition-date groups 隔离，日期不跨 train/validation/test。

这属于完整的 **fold-pure pipeline grouped CV**。NuCLR 与 matcher 分阶段训练并通过冻结 embedding 衔接，
因此不是一个联合反向传播的单体 differentiable end-to-end network；报告中不应混淆这两个概念。

## 文件

- `all_runs.csv`：60 个 Candidate A/B 外层测试结果。
- `aggregate.csv`：折级主汇总。
- `paired_effects.csv`：同 fold/seed 的 B−A 配对效应。
- `seed_summary.csv`：seed 敏感性。
- `nuclr_audit.csv`：30 个 fold-pure Compact NuCLR 审计。
- `raw/`：协议、运行状态与逐运行 summary/config/audit。
- `SHA256SUMS`：提交包完整性校验。
