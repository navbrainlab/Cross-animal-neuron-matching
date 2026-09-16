# Frozen-atlas activity-window stability experiment

This experiment changes only the amount and temporal location of activity
visible for each held-out query animal. Model weights, fold-specific
training-only population atlas, query coordinates, candidate identities and
evaluation identities remain fixed.

Windows use physical seconds because Atanas and Kato/RLD have different
sampling rates. Each short interval is independently linearly resampled to
512 samples, matching the frozen checkpoint input convention. Start, middle
and end windows measure location sensitivity without treating them as extra
cross-validation folds. Reported means/SDs use the five outer folds; metrics
within a fold are first averaged over the three positions.

`summary.csv` is the main result table. `fold_duration_metrics.csv` and
`position_metrics.csv` retain fold/position detail. Each fold directory also
contains exact query predictions (gzip CSV), physical window audits and a
full-record reproduction audit. `minimum_duration.json` reports descriptive
90/95/99% Top-1 retention thresholds relative to the full record; these are
not significance-test claims.
