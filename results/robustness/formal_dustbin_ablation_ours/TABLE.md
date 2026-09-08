# Ours unknown/dustbin robustness and no-dustbin ablation

Kato/RLD grouped CV5 × seed42. Synthetic distractors are unknown positives; atlas-eligible labeled held-out neurons are negatives. Native unlabeled cells remain context but are excluded from binary unknown targets.

Perturbation replicates are averaged within fold first; values are the unweighted mean ± sample SD across five biological folds.

| Transport | Distractor severity | Benchmark Top-1 ↑ | Reject-aware Top-1 ↑ | Unknown Recall ↑ | Forced match ↓ | Dustbin Precision ↑ | Dustbin F1 ↑ | AUROC ↑ | AUPRC ↑ | Known false reject ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| With dustbin | 0.00 | 63.93 ± 4.61% | 57.47 ± 5.47% | N/A | N/A | N/A | N/A | N/A | N/A | 26.52 ± 6.71% |
| With dustbin | 0.10 | 62.74 ± 4.86% | 55.31 ± 5.06% | 90.23 ± 2.82% | 9.77 ± 2.82% | 65.84 ± 5.20% | 76.08 ± 4.36% | 84.59 ± 7.31% | 72.43 ± 12.13% | 29.25 ± 6.00% |
| With dustbin | 0.20 | 61.67 ± 5.58% | 53.07 ± 5.29% | 92.34 ± 1.78% | 7.66 ± 1.78% | 78.18 ± 3.85% | 84.65 ± 2.84% | 83.43 ± 6.55% | 81.91 ± 7.55% | 32.04 ± 6.47% |
| With dustbin | 0.30 | 58.96 ± 5.64% | 49.65 ± 6.00% | 92.52 ± 2.11% | 7.48 ± 2.11% | 82.75 ± 2.65% | 87.35 ± 2.28% | 81.79 ± 6.63% | 85.72 ± 5.50% | 35.71 ± 5.63% |
| With dustbin | 0.40 | 58.40 ± 5.05% | 48.03 ± 4.39% | 93.56 ± 1.70% | 6.44 ± 1.70% | 85.76 ± 1.73% | 89.48 ± 1.54% | 80.97 ± 5.88% | 88.43 ± 3.81% | 38.41 ± 4.89% |
| With dustbin | 0.50 | 56.71 ± 5.82% | 45.29 ± 5.32% | 93.84 ± 1.30% | 6.16 ± 1.30% | 87.58 ± 1.75% | 90.59 ± 1.26% | 80.06 ± 6.17% | 89.98 ± 3.52% | 41.22 ± 6.25% |
| No dustbin | 0.00 | 59.39 ± 6.49% | 59.39 ± 6.49% | N/A | N/A | N/A | N/A | N/A | N/A | 0.00 ± 0.00% |
| No dustbin | 0.10 | 58.20 ± 7.30% | 58.20 ± 7.30% | 0.00 ± 0.00% | 100.00 ± 0.00% | N/A | N/A | N/A | N/A | 0.00 ± 0.00% |
| No dustbin | 0.20 | 56.39 ± 6.92% | 56.39 ± 6.92% | 0.00 ± 0.00% | 100.00 ± 0.00% | N/A | N/A | N/A | N/A | 0.00 ± 0.00% |
| No dustbin | 0.30 | 54.45 ± 7.16% | 54.45 ± 7.16% | 0.00 ± 0.00% | 100.00 ± 0.00% | N/A | N/A | N/A | N/A | 0.00 ± 0.00% |
| No dustbin | 0.40 | 53.93 ± 5.94% | 53.93 ± 5.94% | 0.00 ± 0.00% | 100.00 ± 0.00% | N/A | N/A | N/A | N/A | 0.00 ± 0.00% |
| No dustbin | 0.50 | 51.93 ± 5.86% | 51.93 ± 5.86% | 0.00 ± 0.00% | 100.00 ± 0.00% | N/A | N/A | N/A | N/A | 0.00 ± 0.00% |

`No dustbin` is an inference-only transport ablation: the checkpoint, atlas, node encodings, and relation transport are unchanged. With no rejection state, every synthetic unknown is forcibly assigned to a known identity.
`Benchmark Top-1` ranks only real identity slots, matching the Main Benchmark. `Reject-aware Top-1` additionally counts a known neuron as incorrect when the dustbin is its highest-probability output.
