# Original GeoTransformer: outer-train medoid evaluation

This folder preserves the evaluation-only protocol used to compare the original
semantic GeoTransformer checkpoints with a single training template.

The model, selected checkpoints, dense score construction, Top-1, Top-5, MRR,
Hungarian accuracy, and candidate coverage are unchanged. For each outer fold,
the evaluator selects the training animal with minimum mean symmetric Chamfer
distance to the remaining training animals. It then replaces test-test ordered
pairs by one directional match from every test animal to that fixed template.

Run from the repository root with the environment used for the original jobs:

```bash
python train_medoid_protocol/evaluate_cv5x3_train_medoid.py atanas 2>&1 \
  | tee cv5x3_results_train_medoid_template_v1/atanas.log
python train_medoid_protocol/evaluate_cv5x3_train_medoid.py rld 2>&1 \
  | tee cv5x3_results_train_medoid_template_v1/rld.log
```
