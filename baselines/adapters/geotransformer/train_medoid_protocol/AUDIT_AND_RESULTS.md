# GeoTransformer legacy metric audit and train-medoid results

Date: 2026-08-25

## Provenance of the published local numbers

The local values `31.42 ± 4.66` (Atanas Top-1) and `9.51 ± 1.89`
(RLD Top-1) come from `cv5x3_results/{dataset}` and are reproduced by
`summarize_geotransformer_cv5x3.py`.

The original protocol:

1. Trains seeds 1, 42, and 123 in outer folds 1 through 5.
2. Scans checkpoints on validation and parses the printed, two-decimal Top-1.
   It chooses the highest value; an exact parsed tie chooses the earlier step.
3. Constructs every ordered test-test animal pair with at least five shared
   labelled identities. Both A-to-B and B-to-A are present.
4. Pools query counts across all ordered pairs in each fold/seed cell.
5. Averages the three seed metrics within each fold, then reports the mean and
   sample standard deviation across five fold means.

The model is the full adapted pipeline: KPConv-FPN, official geometric
transformer, coarse correspondence selection, patch matching, and Sinkhorn.
It is not the later simplified constant-feature transformer adapter.

## Legacy metric details

- Top-1 uses `dense.argmax`.
- Top-5 uses `torch.topk`.
- MRR ranks the best correct target with
  `1 + count(candidate_score > correct_score)`.
- Hungarian applies `linear_sum_assignment(-dense)` to the full rectangular
  dense score matrix.
- Metrics are divided by the number of reference identities also found in the
  source animal.
- Candidate coverage records whether the true target survived coarse
  correspondence selection and entered the dense fine-score matrix.

Two legacy caveats remain because the medoid run intentionally preserved the
old metric implementation for exact comparison:

1. Top-1 does not explicitly reject a row for which all entries are the
   missing-candidate sentinel. If column zero happens to be correct, such a row
   can be counted accidentally. Top-5 and MRR do apply candidate-presence
   guards.
2. Hungarian assigns the entire dense matrix, including sentinel entries, and
   also lacks a per-row candidate-presence guard.

## Train-only geometry medoid protocol

For each outer fold, the template is selected only from its training animals.
Each animal is normalized by the original dataset preprocessing (median
centering and 90th-percentile radius scaling). The medoid minimizes its mean
symmetric nearest-neighbour/Chamfer distance to all other training animals.
No validation/test geometry or identity label participates in template
selection. Evaluation direction is test animal to training template.

The model checkpoints, checkpoint selection, dense scores, and legacy metrics
are unchanged.

## Results

Three seeds are averaged within each fold; the final uncertainty is sample SD
across the five fold means.

| Dataset | Top-1 | Top-5 | MRR | Hungarian | Candidate coverage |
|---|---:|---:|---:|---:|---:|
| Atanas | 45.46% ± 5.21% | 73.09% ± 5.25% | 0.5697 ± 0.0527 | 41.40% ± 4.81% | 76.79% ± 6.14% |
| RLD | 25.19% ± 11.26% | 48.29% ± 19.60% | 0.3494 ± 0.1421 | 13.66% ± 6.64% | 60.41% ± 16.64% |

RLD has low effective coverage for some medoids. Fold 1 evaluates 42 shared
identity queries per seed and excludes six test animals with zero shared
identity; fold 4 evaluates 31 queries per seed and excludes one such animal.
Consequently its fold-to-fold uncertainty is large and the mean must not be
interpreted as coverage of every held-out animal.

Complete per-cell counts, templates, checkpoint hashes, and per-test-animal
metrics are stored under
`cv5x3_results_train_medoid_template_v1/{atanas,rld}/REPORT.json`.
