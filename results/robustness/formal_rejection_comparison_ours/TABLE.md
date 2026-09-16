# Validation-calibrated unknown rejection

Kato/RLD grouped CV5 × seed42. Thresholds are calibrated independently per fold using only clean validation known neurons, then frozen for all held-out distractor severities. Native unlabeled cells remain context and are excluded from both known and unknown metrics.

Replicates are averaged within fold first; values are the unweighted mean ± sample SD across five biological folds.

| Setting | Distractor severity | Unknown recall ↑ | Unknown precision ↑ | Known false reject ↓ | Reject-aware Top-1 ↑ | Real-only Top-1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| NeuRID: original (t=0) | 0.00 | N/A | N/A | 26.52 ± 6.71% | 57.47 ± 5.47% | 63.93 ± 4.61% |
| NeuRID: validation-calibrated | 0.00 | N/A | N/A | 6.70 ± 3.69% | 62.75 ± 5.16% | 63.93 ± 4.61% |
| No dustbin + confidence rejection | 0.00 | N/A | N/A | 5.79 ± 3.83% | 58.77 ± 6.28% | 59.39 ± 6.49% |
| No dustbin: forced match | 0.00 | N/A | N/A | 0.00 ± 0.00% | 59.39 ± 6.49% | 59.39 ± 6.49% |
| NeuRID: original (t=0) | 0.10 | 90.23 ± 2.82% | 65.84 ± 5.20% | 29.25 ± 6.00% | 55.31 ± 5.06% | 62.74 ± 4.86% |
| NeuRID: validation-calibrated | 0.10 | 43.71 ± 14.76% | 77.15 ± 13.56% | 7.62 ± 4.71% | 61.63 ± 5.45% | 62.74 ± 4.86% |
| No dustbin + confidence rejection | 0.10 | 49.80 ± 15.64% | 86.25 ± 6.07% | 5.38 ± 3.88% | 57.85 ± 7.21% | 58.20 ± 7.30% |
| No dustbin: forced match | 0.10 | 0.00 ± 0.00% | N/A | 0.00 ± 0.00% | 58.20 ± 7.30% | 58.20 ± 7.30% |
| NeuRID: original (t=0) | 0.20 | 92.34 ± 1.78% | 78.18 ± 3.85% | 32.04 ± 6.47% | 53.07 ± 5.29% | 61.67 ± 5.58% |
| NeuRID: validation-calibrated | 0.20 | 43.18 ± 12.66% | 85.17 ± 7.67% | 8.87 ± 4.37% | 60.40 ± 5.91% | 61.67 ± 5.58% |
| No dustbin + confidence rejection | 0.20 | 53.67 ± 11.67% | 93.73 ± 2.50% | 4.87 ± 3.17% | 56.02 ± 6.98% | 56.39 ± 6.92% |
| No dustbin: forced match | 0.20 | 0.00 ± 0.00% | N/A | 0.00 ± 0.00% | 56.39 ± 6.92% | 56.39 ± 6.92% |
| NeuRID: original (t=0) | 0.30 | 92.52 ± 2.11% | 82.75 ± 2.65% | 35.71 ± 5.63% | 49.65 ± 6.00% | 58.96 ± 5.64% |
| NeuRID: validation-calibrated | 0.30 | 44.92 ± 11.23% | 88.97 ± 5.04% | 10.11 ± 5.09% | 57.35 ± 6.20% | 58.96 ± 5.64% |
| No dustbin + confidence rejection | 0.30 | 55.26 ± 11.20% | 95.80 ± 1.48% | 4.83 ± 2.82% | 54.07 ± 7.02% | 54.45 ± 7.16% |
| No dustbin: forced match | 0.30 | 0.00 ± 0.00% | N/A | 0.00 ± 0.00% | 54.45 ± 7.16% | 54.45 ± 7.16% |
| NeuRID: original (t=0) | 0.40 | 93.56 ± 1.70% | 85.76 ± 1.73% | 38.41 ± 4.89% | 48.03 ± 4.39% | 58.40 ± 5.05% |
| NeuRID: validation-calibrated | 0.40 | 47.09 ± 12.02% | 90.76 ± 3.82% | 11.57 ± 5.07% | 56.29 ± 5.37% | 58.40 ± 5.05% |
| No dustbin + confidence rejection | 0.40 | 56.32 ± 9.81% | 96.73 ± 1.38% | 5.12 ± 3.21% | 53.61 ± 5.75% | 53.93 ± 5.94% |
| No dustbin: forced match | 0.40 | 0.00 ± 0.00% | N/A | 0.00 ± 0.00% | 53.93 ± 5.94% | 53.93 ± 5.94% |
| NeuRID: original (t=0) | 0.50 | 93.84 ± 1.30% | 87.58 ± 1.75% | 41.22 ± 6.25% | 45.29 ± 5.32% | 56.71 ± 5.82% |
| NeuRID: validation-calibrated | 0.50 | 48.05 ± 11.73% | 92.16 ± 3.42% | 12.49 ± 6.11% | 54.68 ± 6.41% | 56.71 ± 5.82% |
| No dustbin + confidence rejection | 0.50 | 55.55 ± 9.41% | 97.46 ± 1.25% | 4.86 ± 3.29% | 51.62 ± 5.72% | 51.93 ± 5.86% |
| No dustbin: forced match | 0.50 | 0.00 ± 0.00% | N/A | 0.00 ± 0.00% | 51.93 ± 5.86% | 51.93 ± 5.86% |

NeuRID rejection uses `dustbin_probability - max_real_probability > t`; the original rule is exactly `t=0`. The confidence baseline rejects when the ordinary no-dustbin transport's maximum real-identity probability is below its validation threshold. Both are applied after relational transport/matching and require no retraining.

Unknown precision is N/A when a rule predicts no rejections (including forced matching). Test false-reject rates are measured, not imputed from the 5% validation budget.

