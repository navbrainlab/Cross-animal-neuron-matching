# NeuRID 可视化用逐神经元导出

文件：`neurid_query_neurons_atanas_cv5_seed42.csv`

这是 Atanas/SF 000776 数据集正式 grouped CV5、seed 42、held-out test 的导出。折号沿用代码中的零基编号 `0`–`4`。CSV 覆盖全部 38 个测试动物，共 5,213 个神经元；其中 3,210 个属于正式评估集合。

## 字段与 Top-1 口径

- `neuron_row` 是对应原始 NPZ 记录的零基行号；`fold + animal_id + neuron_row` 唯一。
- `gt_identity` 直接使用源记录的 `cell_id`。无标注值留空；带 `?` 的不确定源标注予以保留，但不属于正式评估集合。
- `pred_geometry` 是正式 `geometry_only`（论文中的 **w/o Activity**）模型的预测；该变体同时移除 activity node 和 activity relation 输入。
- `pred_full` 是正式 Full NeuRID 模型的预测。
- 两个预测均为静态 atlas 输出 `row_conditional[:, :-1]` 中概率最高的**真实身份候选**，即正式 `top1_real` 的定义。它们不是 Hungarian 结果，并且即使 dustbin 分数更高，也仍返回最高分真实候选。
- `is_evaluated=true` 精确复用正式评估导出的 query 集合：源标注必须通过 `labeled/certain/clean` 掩码、在该动物内唯一，并且存在于该折的训练 atlas 候选中。其余行仍保留两模型预测，仅用于空间背景或探索，不计入正式指标。

正式评估行的预测槽位已逐行与归档结果核对，Full 与 w/o Activity 均完全一致。

## 坐标

CSV 中的 `x, y, z` 是源 NPZ 的未归一化 `xyz` 物理坐标，单位为 µm；它由 ROI 的 voxel centroid 乘以该记录的 `grid_spacing` 得到。导出值不是模型内部坐标。NeuRID 在编码时会对每个动物独立做按轴的 median centering 和 RMS scale normalization。

## 两个模型的可比性

两个模型使用完全相同的测试动物、测试神经元、正式评估 query 集合、fold 划分以及静态 atlas 构建协议。每个模型均只从同一折的训练动物和同一套训练标注掩码构建自己的模型特定 latent atlas；测试动物不参与 atlas 构建。

每折两个模型的候选身份到 atlas slot 的映射完全相同。候选集合由该折训练数据决定，因此折间大小略有不同：

| fold | 测试动物 | 全部神经元 | 正式 query | 候选身份数 | w/o Activity Top-1 | Full Top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 1,099 | 664 | 152 | 34.34% | 70.78% |
| 1 | 8 | 1,103 | 669 | 152 | 51.27% | 80.27% |
| 2 | 8 | 1,076 | 656 | 152 | 41.16% | 62.35% |
| 3 | 7 | 961 | 610 | 157 | 45.90% | 79.02% |
| 4 | 7 | 974 | 611 | 156 | 43.54% | 82.16% |

身份已直接写成神经元名称（例如 `AVAL`、`RMDL`），没有数字编码，因此不需要另附数字映射表。

## 完整性信息

- 非空源身份：3,397 行
- 未标注身份：1,816 行
- 两列预测均非空：5,213 行
- CSV SHA-256：`11ea31c52bd72786692416aba668c549fd9ba024309e952c5d70bbcf8d05af74`

可从仓库根目录重建并执行全部一致性检查：

```bash
python scripts/neurid/export_atanas_visualization_csv.py
```
