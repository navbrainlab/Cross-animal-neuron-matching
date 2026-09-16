# Published results

This directory contains the frozen tables, fold-level metrics, query-level
predictions, audits, and figure data used by the manuscript. Start with
[`PAPER_RESULTS_MANIFEST.md`](PAPER_RESULTS_MANIFEST.md) to trace each paper
table or figure to both its result files and its code.

The headline seed-42 results are:

- Atanas Full Top-1: **74.92 ± 8.26%**
- Kato Full Top-1: **63.93 ± 4.61%**
- Atanas Shared Top-1: **79.01 ± 8.84%**
- Kato Shared Top-1: **74.78 ± 5.24%**
- Zebrafish LOFO8 Top-1: **97.69 ± 3.84%**

All means and sample standard deviations use biological folds as the unit of
aggregation. Result files are immutable evidence; new executions belong in
the Git-ignored `runs/` directory.
