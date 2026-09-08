# 训练集几何 medoid 模板公平重评估

## 固定协议

- 每个 outer fold 只在该折 `train/` 内选择一个固定模板；`val/`、`test/` 均不参与模板选择。
- 所有训练虫先按现有协议逐虫做 XYZ 中位数中心化并除以 200。两虫几何距离定义为 symmetric Chamfer：两个方向的平均最近邻欧氏距离再取均值。
- 对每只训练虫，计算它到其余所有训练虫的平均几何距离；选择均值最小者作为 medoid。完全相同时按 worm UID、train-list index 确定性打破并列。
- medoid 选择不读取 `cell_id`。候选身份词表只来自选中的训练模板，不再使用训练虫身份并集。
- CPD、fDNC、NuCLR、GeoTransformer 以及兼容的 MPRT/FGW/RGM checkpoint 都将每只 validation/test 虫独立匹配到同一个训练 medoid；禁止 test–test 两两匹配。GeoTransformer 使用独立保存的完整官方适配流水线，不使用已经删除的简化 adapter。
- 主分析对“测试神经元→medoid 模板神经元”的分数做 row-z-score。只有一个模板，不再进行跨训练虫均值或投票。
- Ours 直接匹配 checkpoint 内由训练集构建并带固定身份语义的 Atlas。
- 测试标签只在最终计分阶段读取。test-only 身份不在训练词表中，记为不可评估，不可临时加入候选集。
- 统一报告 Top-1、Top-5、MRR 和 Hungarian accuracy；ranking 并列按随机均匀打破并列时的期望值计分。
- 多随机种子先在同一 fold 内平均，再报告 5 个 outer folds 的非加权均值和样本标准差。

核心实现是 `scripts/lib/fair_identity_protocol.py`；固定 medoid 模板重前向入口是
`scripts/fair_identity/evaluate_train_reference_ensemble.py`，支持 `cpd`、`fdnc`、预提取 `nuclr`、
`mprt`（可用于兼容的 FGW/RGM checkpoint）。GeoTransformer 的独立入口是
`../geotransformer_official/train_medoid_protocol/evaluate_cv5x3_train_medoid.py`。
NuCLR 的 train/test embedding 必须由该 fold 的同一训练 checkpoint 提取，并输出
相同的 `[test_neuron, reference_neuron]` 分数矩阵，
再调用同一核心，不能继续汇总旧的 test-to-test 或全训练参考 ensemble 指标。

## 旧结果状态

以下 CPD/fDNC 表格是此前“匹配全部训练虫并聚合”的历史结果，仅保留溯源，
**不属于当前 medoid-template 协议，不能进入新主表**。切换协议后必须逐 fold 重跑。

## 历史 CV5 × 3-seed 结果（仅 provenance / supplementary stability）

以下 Ours 汇总来自 seeds 1/42/123 先折内平均的旧口径。它们可用于历史溯源或明确标注
的 supplementary stability check，但不再是正式主表结果。正式主 Benchmark 固定为同一
CV5 × seed42；Ours Top-1 为 Atanas **74.92 ± 8.26%**、RLD **63.93 ± 4.61%**，详见
`runs/fair_identity_seed42_benchmark_v1/BENCHMARK_SEED42.md`。

| Dataset | Method | Top-1 | Top-5 | MRR | Hungarian |
| --- | --- | ---: | ---: | ---: | ---: |
| Atanas | Ours dynamic Atlas | 74.45 ± 7.30 | 90.25 ± 5.66 | 81.56 ± 6.42 | 74.39 ± 7.38 |
| Atanas | Ours static Atlas | 74.24 ± 7.40 | 90.06 ± 5.82 | 81.35 ± 6.42 | 74.07 ± 7.42 |
| RLD | Ours dynamic Atlas | 63.32 ± 3.81 | 80.12 ± 3.86 | 71.14 ± 3.30 | 61.50 ± 3.92 |
| RLD | Ours static Atlas | 63.14 ± 3.52 | 80.10 ± 3.82 | 70.91 ± 3.10 | 61.22 ± 3.58 |

表中均为 5 folds 的百分数 mean ± sample SD；每折已先平均 seeds 1/42/123。

各 pairwise 方法的旧汇总是同 split 动物两两匹配或固定单参考，不能数学上转换为
全训练参考身份平均。除下方已经重跑的 CPD 和官方原始 fDNC 外，NuCLR、
GeoTransformer 和 Vanilla FGW/RGM 只有在逐对重前向完成后才可进入同一比较表；
此前应标为 pending。

## CPD 全量重跑结果

CPD 已完成并在统计中使用 Atanas 865 个和 RLD 5,301 个有向配准矩阵。RLD 中两个
测试动物没有任何 clean GT 身份进入相应训练词表，按协议列为不可评估；一次边界检查
还缓存了其中一只动物的 57 个矩阵，但未纳入指标。最终 query 数 1,457，与 Ours Atlas
评估完全一致。

| Dataset | 汇总规则 | Top-1 | Top-5 | MRR | Hungarian |
| --- | --- | ---: | ---: | ---: | ---: |
| Atanas | row-z-score 后同身份均值（主） | 27.61 ± 2.77 | 69.84 ± 3.54 | 45.69 ± 2.72 | 44.15 ± 3.39 |
| Atanas | 训练参考投票 | 47.81 ± 3.80 | 81.73 ± 5.29 | 62.31 ± 4.32 | 52.14 ± 3.94 |
| RLD | row-z-score 后同身份均值（主） | 0.89 ± 0.51 | 4.60 ± 1.06 | 5.22 ± 0.98 | 1.69 ± 1.02 |
| RLD | 训练参考投票 | 17.67 ± 1.54 | 46.24 ± 2.67 | 31.81 ± 1.50 | 26.11 ± 2.22 |

均值规则是主输出，投票是敏感性分析，不按测试集表现二选一。无温度校准的
row-softmax 均值另行保留为诊断，但不作为主表；二者差距说明 CPD 的跨参考分数聚合
尤其在 RLD 上不稳定，论文中应如实报告该敏感性。

## fDNC 全量重跑结果

fDNC 使用官方原始 `model.bin`（SHA256
`ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9`），不使用与当前
`cv5_grouped_v1` 不同划分上微调过的旧 fold checkpoint，避免 outer-test 泄漏。模型
权重固定，仅将每个测试动物分别与当前折的全部训练动物匹配。共使用 Atanas 865 个和
RLD 5,301 个有向分数矩阵；可评估 query 数分别为 3,210 和 1,457。RLD 两个无训练
词表身份重叠的测试动物按固定协议跳过。

| Dataset | 汇总规则 | Top-1 | Top-5 | MRR | Hungarian |
| --- | --- | ---: | ---: | ---: | ---: |
| Atanas | row-z-score 后同身份均值（主） | 10.44 ± 3.84 | 45.86 ± 9.53 | 26.89 ± 6.02 | 22.20 ± 5.08 |
| Atanas | 训练参考投票 | 20.93 ± 4.74 | 53.34 ± 10.05 | 35.42 ± 6.65 | 23.92 ± 6.03 |
| RLD | row-z-score 后同身份均值（主） | 1.70 ± 0.93 | 14.48 ± 2.46 | 10.53 ± 1.26 | 3.62 ± 1.22 |
| RLD | 训练参考投票 | 18.02 ± 2.56 | 46.32 ± 2.87 | 31.24 ± 2.34 | 26.59 ± 2.45 |

与 CPD 相同，row-z-score 身份均值是预先固定的主输出，投票只作为敏感性分析，不能
根据测试集上哪一个更高来选择。完整逐折结果与汇总位于
`runs/fair_identity_retest_v1/fdnc_official/`。

## 运行示例

```bash
python -m scripts.fair_identity.evaluate_train_reference_ensemble \
  --method cpd \
  --fold-root Data/Atanas_SF_unified_000776/cv5_grouped_v1/fold_0 \
  --cache-dir runs/fair_identity_medoid_template_v1/cache/cpd/atanas/fold0 \
  --output-dir runs/fair_identity_medoid_template_v1/cpd/atanas/fold0

cd ../geotransformer_official
/home/ubuntu/anaconda3/envs/nuclr310/bin/python \
  train_medoid_protocol/evaluate_cv5x3_train_medoid.py atanas
```
