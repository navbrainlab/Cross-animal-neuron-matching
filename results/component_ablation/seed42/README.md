# Formal component-ablation evidence

This directory is the self-contained audit record for the four ablation rows
in the Atanas and Kato/RLD table. The formal protocol is five grouped
biological folds and model seed 42. Means and sample standard deviations are
computed across the five fold-level values.

Files:

- `fold_cells.csv`: held-out-test metrics for Full and all four ablations,
  one row per dataset, fold, and arm (50 rows).
- `query_results.csv`: every held-out test query for the four ablations
  (18,668 query-arm rows). It contains ranks, correctness, target
  probabilities, dustbin decisions, and Hungarian predictions. Absolute local
  paths have been replaced by recording basenames.
- `summary.csv` and `SUMMARY.md`: recomputed five-fold summaries.
- `VALIDATION.json`: machine-readable audit result and input hashes.
- `PROTOCOL.json`: protocol and exact internal-arm-to-paper-row mapping.

Run the independent reconstruction from the repository root:

```bash
python scripts/neurid/summarize_component_ablation_seed42.py
```

The command fails closed on missing/duplicate cells, cohort differences among
the four ablations, query-to-fold metric mismatches, a non-test/non-seed-42
row, or a mismatch between Full and the archived main benchmark.

## Manuscript Top-1 check

| Variant | Atanas | Kato/RLD | Check |
|---|---:|---:|:---:|
| w/o Population Relations | 62.93 ± 7.31% | 33.95 ± 3.84% | PASS |
| w/o Relation Transport | 73.16 ± 6.82% | 55.91 ± 5.06% | PASS |
| w/o Geometry | 41.07 ± 4.75% | 20.76 ± 3.22% | PASS |
| w/o Activity | 43.24 ± 6.23% | 44.76 ± 4.07% | PASS |

These four manuscript rows require no numerical change under the formal
CV5 × seed-42 protocol.
