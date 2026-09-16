# Frozen Full NeuRID representation — matcher controls

Unweighted mean and sample sd over five biological folds; all matchers use the same frozen contextual embeddings and train-only atlas.

| Dataset | Matcher | Top-1 | Top-5 | MRR | Hungarian | Partial assignment | Novel reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| atanas | Sinkhorn | 62.99 ± 6.10% | 85.68 ± 5.35% | 0.7312 ± 0.0539 | 63.13 ± 5.65% | 62.64 ± 6.28% | 0.00 ± 0.00% |
| atanas | FGW | 51.44 ± 4.94% | 64.94 ± 6.35% | 0.5866 ± 0.0550 | 51.47 ± 4.92% | 51.15 ± 5.11% | 0.00 ± 0.00% |
| atanas | UFGW | 67.85 ± 7.09% | 87.58 ± 5.11% | 0.7665 ± 0.0604 | 67.64 ± 6.36% | 67.48 ± 7.23% | 0.00 ± 0.00% |
| atanas | NeuRID | 74.92 ± 8.26% | 90.33 ± 5.69% | 0.8186 ± 0.0687 | 75.00 ± 8.37% | 73.39 ± 8.62% | 16.67 ± 23.57% |
| rld | Sinkhorn | 58.22 ± 6.66% | 77.46 ± 3.54% | 0.6720 ± 0.0514 | 55.50 ± 5.47% | 55.79 ± 7.99% | 0.57 ± 1.28% |
| rld | FGW | 47.24 ± 4.49% | 63.86 ± 5.55% | 0.5588 ± 0.0473 | 47.24 ± 4.49% | 45.18 ± 5.69% | 0.00 ± 0.00% |
| rld | UFGW | 56.76 ± 5.34% | 77.23 ± 3.93% | 0.6643 ± 0.0469 | 54.56 ± 5.72% | 54.29 ± 6.80% | 0.00 ± 0.00% |
| rld | NeuRID | 63.93 ± 4.61% | 79.81 ± 5.01% | 0.7137 ± 0.0438 | 61.78 ± 5.18% | 58.22 ± 4.66% | 66.83 ± 11.94% |
