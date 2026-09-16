# Hierarchical, validation-calibrated unknown rejection

Kato/RLD grouped CV5 × seed42. One threshold is selected per fold from pooled validation distractor levels and frozen across all held-out test levels. No model is retrained.

Replicates are averaged within fold first; values are the unweighted mean ± sample SD across five biological folds. Confidence intervals for differences are two-sided paired Student-t intervals across the same folds.

## Clean known-only condition (`r_dist = 0`)

| Rejection rule | Known false reject ↓ | Reject-aware Top-1 ↑ |
|---|---:|---:|
| Current: p⊥ > pmax | 26.52 ± 6.71% | 57.47 ± 5.47% |
| Binary: p⊥ > 0.5 | 18.66 ± 7.81% | 59.45 ± 5.70% |
| Val p⊥ @ 80% recall | 20.51 ± 8.48% | 58.72 ± 5.50% |
| Val p⊥ @ 90% recall | 27.89 ± 9.15% | 55.71 ± 6.14% |
| Val p⊥ @ 95% recall | 34.87 ± 9.15% | 52.39 ± 6.20% |
| Val log-margin @ 80% recall | 15.23 ± 6.41% | 60.91 ± 5.11% |
| Val log-margin @ 90% recall | 22.48 ± 6.26% | 58.79 ± 5.06% |
| Val log-margin @ 95% recall | 29.33 ± 7.46% | 56.46 ± 5.55% |

On clean data, the 80%-targeted log-margin rule reduced known false rejection from 26.52% to 15.23%, a paired reduction of 11.29 pp (95% CI [8.33, 14.24]).

## Unknown-present conditions (pooled `r_dist = 0.1–0.5`)

| Rejection rule | Unknown recall ↑ | Known false reject ↓ | Reject-aware Top-1 ↑ | Real-only Top-1 ↑ |
|---|---:|---:|---:|---:|
| Current: p⊥ > pmax | 92.50 ± 1.86% | 35.32 ± 5.73% | 50.27 ± 5.16% | 59.70 ± 5.35% |
| Binary: p⊥ > 0.5 | 74.93 ± 9.25% | 25.57 ± 7.44% | 53.06 ± 5.64% | 59.70 ± 5.35% |
| Val p⊥ @ 80% recall | 79.12 ± 3.49% | 28.41 ± 8.64% | 51.79 ± 6.31% | 59.70 ± 5.35% |
| Val p⊥ @ 90% recall | 88.93 ± 1.97% | 38.21 ± 8.73% | 47.32 ± 6.22% | 59.70 ± 5.35% |
| Val p⊥ @ 95% recall | 94.27 ± 1.43% | 46.83 ± 8.21% | 42.62 ± 5.84% | 59.70 ± 5.35% |
| Val log-margin @ 80% recall | 78.46 ± 3.49% | 20.53 ± 6.39% | 55.88 ± 5.80% | 59.70 ± 5.35% |
| Val log-margin @ 90% recall | 88.88 ± 2.09% | 29.62 ± 5.93% | 52.65 ± 5.45% | 59.70 ± 5.35% |
| Val log-margin @ 95% recall | 94.25 ± 1.26% | 38.65 ± 5.71% | 48.65 ± 4.96% | 59.70 ± 5.35% |

### Paired effect of the 90%-targeted log-margin rule

Relative to the default rule, it reduced known false rejection by 5.70 pp (95% CI [1.11, 10.29]) and increased reject-aware Top-1 by 2.39 pp (95% CI [0.27, 4.50]).

## Per-severity unknown-present results

| Rejection rule | Distractor fraction | Unknown recall ↑ | Known false reject ↓ | Reject-aware Top-1 ↑ | Real-only Top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| Current: p⊥ > pmax | 0.10 | 90.23 ± 2.82% | 29.25 ± 6.00% | 55.31 ± 5.06% | 62.74 ± 4.86% |
| Binary: p⊥ > 0.5 | 0.10 | 70.83 ± 11.38% | 20.25 ± 7.37% | 57.67 ± 5.53% | 62.74 ± 4.86% |
| Val p⊥ @ 80% recall | 0.10 | 76.15 ± 5.79% | 22.77 ± 8.36% | 56.56 ± 5.59% | 62.74 ± 4.86% |
| Val p⊥ @ 90% recall | 0.10 | 86.81 ± 3.56% | 31.15 ± 8.77% | 53.38 ± 5.86% | 62.74 ± 4.86% |
| Val p⊥ @ 95% recall | 0.10 | 92.75 ± 2.63% | 38.49 ± 8.89% | 49.37 ± 5.95% | 62.74 ± 4.86% |
| Val log-margin @ 80% recall | 0.10 | 75.44 ± 4.58% | 16.26 ± 5.98% | 60.10 ± 5.44% | 62.74 ± 4.86% |
| Val log-margin @ 90% recall | 0.10 | 86.20 ± 3.32% | 24.31 ± 6.71% | 57.22 ± 5.09% | 62.74 ± 4.86% |
| Val log-margin @ 95% recall | 0.10 | 92.44 ± 2.15% | 32.02 ± 6.55% | 54.08 ± 4.96% | 62.74 ± 4.86% |
| Current: p⊥ > pmax | 0.20 | 92.34 ± 1.78% | 32.04 ± 6.47% | 53.07 ± 5.29% | 61.67 ± 5.58% |
| Binary: p⊥ > 0.5 | 0.20 | 73.79 ± 9.08% | 22.55 ± 7.14% | 55.72 ± 5.80% | 61.67 ± 5.58% |
| Val p⊥ @ 80% recall | 0.20 | 78.18 ± 4.16% | 25.44 ± 8.48% | 54.43 ± 6.01% | 61.67 ± 5.58% |
| Val p⊥ @ 90% recall | 0.20 | 87.85 ± 2.28% | 34.58 ± 9.40% | 50.42 ± 6.37% | 61.67 ± 5.58% |
| Val p⊥ @ 95% recall | 0.20 | 93.93 ± 1.49% | 42.51 ± 8.66% | 46.23 ± 5.84% | 61.67 ± 5.58% |
| Val log-margin @ 80% recall | 0.20 | 76.96 ± 4.34% | 18.21 ± 6.20% | 58.19 ± 5.71% | 61.67 ± 5.58% |
| Val log-margin @ 90% recall | 0.20 | 88.43 ± 2.17% | 26.52 ± 6.30% | 55.22 ± 5.41% | 61.67 ± 5.58% |
| Val log-margin @ 95% recall | 0.20 | 93.97 ± 1.14% | 35.15 ± 6.38% | 51.71 ± 5.08% | 61.67 ± 5.58% |
| Current: p⊥ > pmax | 0.30 | 92.52 ± 2.11% | 35.71 ± 5.63% | 49.65 ± 6.00% | 58.96 ± 5.64% |
| Binary: p⊥ > 0.5 | 0.30 | 75.03 ± 8.75% | 25.99 ± 8.04% | 52.35 ± 6.46% | 58.96 ± 5.64% |
| Val p⊥ @ 80% recall | 0.30 | 78.75 ± 3.67% | 28.55 ± 9.41% | 51.22 ± 7.43% | 58.96 ± 5.64% |
| Val p⊥ @ 90% recall | 0.30 | 88.77 ± 2.13% | 38.22 ± 8.93% | 46.94 ± 7.06% | 58.96 ± 5.64% |
| Val p⊥ @ 95% recall | 0.30 | 94.19 ± 1.57% | 46.91 ± 8.81% | 42.15 ± 6.59% | 58.96 ± 5.64% |
| Val log-margin @ 80% recall | 0.30 | 77.75 ± 3.79% | 20.73 ± 6.70% | 55.18 ± 6.31% | 58.96 ± 5.64% |
| Val log-margin @ 90% recall | 0.30 | 88.95 ± 2.17% | 30.23 ± 6.03% | 52.11 ± 6.37% | 58.96 ± 5.64% |
| Val log-margin @ 95% recall | 0.30 | 94.16 ± 1.52% | 39.17 ± 6.04% | 47.77 ± 5.72% | 58.96 ± 5.64% |
| Current: p⊥ > pmax | 0.40 | 93.56 ± 1.70% | 38.41 ± 4.89% | 48.03 ± 4.39% | 58.40 ± 5.05% |
| Binary: p⊥ > 0.5 | 0.40 | 76.93 ± 9.15% | 28.23 ± 7.18% | 50.96 ± 4.88% | 58.40 ± 5.05% |
| Val p⊥ @ 80% recall | 0.40 | 80.74 ± 4.23% | 31.35 ± 8.29% | 49.50 ± 5.84% | 58.40 ± 5.05% |
| Val p⊥ @ 90% recall | 0.40 | 90.01 ± 2.18% | 42.05 ± 8.38% | 44.42 ± 6.12% | 58.40 ± 5.05% |
| Val p⊥ @ 95% recall | 0.40 | 94.96 ± 1.03% | 51.29 ± 7.75% | 39.27 ± 5.82% | 58.40 ± 5.05% |
| Val log-margin @ 80% recall | 0.40 | 80.61 ± 3.61% | 23.02 ± 6.28% | 53.88 ± 5.31% | 58.40 ± 5.05% |
| Val log-margin @ 90% recall | 0.40 | 90.06 ± 2.12% | 32.59 ± 5.14% | 50.47 ± 4.87% | 58.40 ± 5.05% |
| Val log-margin @ 95% recall | 0.40 | 95.26 ± 0.88% | 42.07 ± 4.90% | 46.17 ± 4.29% | 58.40 ± 5.05% |
| Current: p⊥ > pmax | 0.50 | 93.84 ± 1.30% | 41.22 ± 6.25% | 45.29 ± 5.32% | 56.71 ± 5.82% |
| Binary: p⊥ > 0.5 | 0.50 | 78.06 ± 9.04% | 30.84 ± 8.10% | 48.61 ± 5.83% | 56.71 ± 5.82% |
| Val p⊥ @ 80% recall | 0.50 | 81.81 ± 3.29% | 33.93 ± 8.80% | 47.24 ± 6.88% | 56.71 ± 5.82% |
| Val p⊥ @ 90% recall | 0.50 | 91.19 ± 1.30% | 45.04 ± 8.61% | 41.44 ± 6.06% | 56.71 ± 5.82% |
| Val p⊥ @ 95% recall | 0.50 | 95.50 ± 0.78% | 54.96 ± 7.42% | 36.10 ± 5.58% | 56.71 ± 5.82% |
| Val log-margin @ 80% recall | 0.50 | 81.53 ± 2.87% | 24.45 ± 6.94% | 52.02 ± 6.47% | 56.71 ± 5.82% |
| Val log-margin @ 90% recall | 0.50 | 90.76 ± 1.56% | 34.46 ± 5.84% | 48.26 ± 5.76% | 56.71 ± 5.82% |
| Val log-margin @ 95% recall | 0.50 | 95.44 ± 0.95% | 44.85 ± 5.08% | 43.54 ± 4.89% | 56.71 ± 5.82% |

For every non-rejected neuron, identity is decoded as the highest-probability real candidate. `p⊥` thresholds answer known-vs-unknown independently of the number of real identity classes. The log-margin rules are reported only as a robustness check.

The operating curves are descriptive held-out-test curves. Every marked validation operating point was selected without access to test labels.

At 50% distractors, the capacity-dustbin ranking gain over forced no-dustbin matching is 4.78 points. The 80%-recall probability rule costs 9.47 points, whereas the log-margin robustness rule costs 4.69 points and retains a +0.09-point net gain.
