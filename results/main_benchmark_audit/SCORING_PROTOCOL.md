# Scoring contract and tie semantics

## Shared denominator

For fold `f`, the denominator is the canonical test-query manifest: unique, valid GT neurons whose identity occurs in the strict outer-training identity union. The query set is method-independent. If a single-medoid reference lacks the GT identity, all recognition metrics receive zero credit for that query.

## Cross-fold summary

Metrics are computed within each biological fold. The table reports the unweighted arithmetic mean across five folds and the sample standard deviation (`ddof=1`). It is not a pooled-neuron estimate and not a ratio of fold means.

## Current saved Top-k semantics

- NGM-v2, NeurID, GeoTransformer, and NuCLR use competition rank: `1 + count(score > GT score)`. Exact ties at the cutoff are therefore tie-inclusive.
- CPD/fDNC use fractional expected credit within an exact tie block; their `rank_min` and `rank_max` are saved.
- Vanilla FGW uses stable reference-column ordering, matching its official evaluator, because the transport plan contains many exact zeros.
- The main table is not silently changed by this audit. Method-specific historical tie semantics are disclosed here. A future table-wide tie-policy migration requires score matrices for every method and must be versioned as a new result.

## NGM-v2 tie audit

`tie_diagnostics.csv` gives, per query, the number of strictly larger and exactly equal candidate scores, rank interval, reported competition credit, fractional-tie credit, and stable Top-5 identities. `ngmv2_top5_tie_summary.csv` reports the resulting fold-level sensitivity on the canonical denominator.
