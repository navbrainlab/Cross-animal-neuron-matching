# Activity-noise robustness — RLD grouped CV5 × seed42

Noise is added only to held-out-test `activity_raw` as independent Gaussian noise with standard deviation `severity × each neuron's temporal SD`. Three corruption draws are averaged within each fold, followed by the unweighted mean ± sample SD across the same five biological folds. Severity 0 must reproduce each method's formal Main Benchmark cells.

| Method | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| NuCLR | 0.00 | 10.14 ± 7.62% | 13.16 ± 3.62% | 100.00 ± 0.00% | 10.14 ± 7.62% |
| NuCLR | 0.10 | 8.76 ± 4.95% | 10.91 ± 4.35% | 100.00 ± 0.00% | 8.76 ± 4.95% |
| NuCLR | 0.20 | 6.11 ± 3.40% | 6.84 ± 3.59% | 100.00 ± 0.00% | 6.11 ± 3.40% |
| NuCLR | 0.50 | 1.54 ± 2.36% | 4.05 ± 1.52% | 100.00 ± 0.00% | 1.54 ± 2.36% |
| NuCLR | 1.00 | 0.00 ± 0.00% | 3.20 ± 0.89% | 100.00 ± 0.00% | 0.00 ± 0.00% |
| NuCLR | 2.00 | 0.00 ± 0.00% | 2.19 ± 1.05% | 100.00 ± 0.00% | 0.00 ± 0.00% |
| GWOT-MD | — | N/A | N/A | N/A | N/A |
| Vanilla FGW | 0.00 | 7.99 ± 2.44% | 7.85 ± 2.52% | 100.00 ± 0.00% | 7.99 ± 2.44% |
| Vanilla FGW | 0.10 | 7.99 ± 2.44% | 7.85 ± 2.52% | 100.00 ± 0.00% | 7.99 ± 2.44% |
| Vanilla FGW | 0.20 | 7.99 ± 2.44% | 7.85 ± 2.52% | 100.00 ± 0.00% | 7.99 ± 2.44% |
| Vanilla FGW | 0.50 | 7.99 ± 2.44% | 7.85 ± 2.52% | 100.00 ± 0.00% | 7.99 ± 2.44% |
| Vanilla FGW | 1.00 | 7.99 ± 2.44% | 7.85 ± 2.52% | 100.00 ± 0.00% | 7.99 ± 2.44% |
| Vanilla FGW | 2.00 | 7.99 ± 2.44% | 7.85 ± 2.52% | 100.00 ± 0.00% | 7.99 ± 2.44% |
| Ours | 0.00 | 63.93 ± 4.61% | 61.78 ± 5.18% | 100.00 ± 0.00% | 63.93 ± 4.61% |
| Ours | 0.10 | 62.95 ± 4.85% | 61.46 ± 4.62% | 100.00 ± 0.00% | 62.95 ± 4.85% |
| Ours | 0.20 | 62.07 ± 3.98% | 60.09 ± 4.21% | 100.00 ± 0.00% | 62.07 ± 3.98% |
| Ours | 0.50 | 48.28 ± 6.56% | 46.51 ± 6.38% | 100.00 ± 0.00% | 48.28 ± 6.56% |
| Ours | 1.00 | 26.30 ± 4.06% | 23.91 ± 3.63% | 100.00 ± 0.00% | 26.30 ± 4.06% |
| Ours | 2.00 | 15.18 ± 3.62% | 14.25 ± 3.12% | 100.00 ± 0.00% | 15.18 ± 3.62% |

Vanilla FGW is flat by construction in the locked formal implementation: it consumes geometry only. Its values are backed by a 1,520-file input-invariance audit rather than imputed from an unrelated run.

GWOT-MD is intentionally N/A: no canonical RLD held-out-test Main Benchmark cells exist, so severity=0 cannot be locked. Historical GWOT-MD files are validation-only and are not substituted.
