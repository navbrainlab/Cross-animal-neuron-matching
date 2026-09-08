# Zebrafish NeuRID: 8-fold LOFO, seed42

Primary point estimate averages the 8 held-out fish. The ± term below is the SD across held-out fish; 95% CIs bootstrap held-out fish.

| Method | Top-1 | Top-5 | MRR | Hungarian | Hierarchical 95% CI (Top-1) |
|---|---:|---:|---:|---:|---:|
| Euclidean | 75.29% ± 10.45% | 99.07% | 0.8572 ± 0.0693 | 92.43% ± 6.40% | [68.04%, 81.26%] |
| Full NeuRID | 97.69% ± 3.84% | 99.90% | 0.9867 ± 0.0238 | 97.62% ± 3.97% | [94.89%, 99.35%] |
| No relation transport | 97.32% ± 2.24% | 99.86% | 0.9848 | 97.25% | [95.70%, 98.56%] |
| Geometry-only | 96.03% ± 6.10% | 99.89% | 0.9772 | 96.05% | [91.68%, 98.98%] |
| Activity-only | 8.91% ± 3.59% | 30.64% | 0.2097 | 8.70% | [6.70%, 11.30%] |

## Paired Top-1 contrasts

- Full_minus_NoTransport: +0.37 pp, 95% CI [-1.06, +1.43], run wins 5/8, fish wins 5/8, exact sign-flip p=0.6562.
- Full_minus_GeometryOnly: +1.66 pp, 95% CI [+0.10, +3.66], run wins 4/8, fish wins 4/8, exact sign-flip p=0.1250.
- Full_minus_ActivityOnly: +88.78 pp, 95% CI [+85.39, +91.79], run wins 8/8, fish wins 8/8, exact sign-flip p=0.0078.

All eight Full/seed42 row-permutation audits passed.

Task interpretation: unseen-fish longitudinal matching of the same tracked neurons, not a shared cross-fish identity atlas.
