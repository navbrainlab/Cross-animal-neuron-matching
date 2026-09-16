# Atlas relation support ablation

Locked seed-42 CV5 inference-only comparison. Values are the unweighted mean ± sample SD across five folds on the existing training-atlas-identity query universe. No model was retrained or selected for this intervention.

| Atlas relation handling | Atanas Top-1 | Kato/RLD Top-1 |
| --- | ---: | ---: |
| Zero fill | 74.92 ± 8.26% | 63.93 ± 4.61% |
| Mask missing relations | 73.65 ± 8.75% | 60.22 ± 5.44% |

The mask is `M[j,l] = 1[pair_count[j,l] > 0]`; the discrepancy is divided by supported transport mass. Ordered relation entries, including the diagonal, are counted because the learned relation field is directed.

| Dataset | Fold | Unobserved | All | Unobserved fraction |
| --- | ---: | ---: | ---: | ---: |
| Atanas | 0 | 1948 | 23104 | 8.43% |
| Atanas | 1 | 1752 | 23104 | 7.58% |
| Atanas | 2 | 1582 | 23104 | 6.85% |
| Atanas | 3 | 2308 | 24649 | 9.36% |
| Atanas | 4 | 2382 | 24336 | 9.79% |
| Kato/RLD | 0 | 8082 | 15376 | 52.56% |
| Kato/RLD | 1 | 12532 | 21025 | 59.61% |
| Kato/RLD | 2 | 11912 | 20449 | 58.25% |
| Kato/RLD | 3 | 12202 | 21025 | 58.04% |
| Kato/RLD | 4 | 6786 | 14161 | 47.92% |
