# Known limitations and provenance boundaries

- CPD and fDNC save the Top-1 predicted identity and rank/Top-k correctness, but not the ordered Top-5 identities. Their Hungarian result is saved only as a fold-level numerator/accuracy, not as a per-query assignment.
- GeoTransformer and Vanilla FGW save Top-1 identity plus rank, Top-5 correctness, and Hungarian correctness; they do not save the ordered Top-5 identities or Hungarian-predicted identity.
- NGM-v2 complete score matrices and candidate order are included, but they are deterministic replays of the frozen checkpoints rather than files emitted by the first evaluation. The row-by-row exact replay guard is included in every fold metrics JSON.
- Existing NuCLR and both NeurID result CSVs save ranks/correctness but not decoded prediction identities. NuCLR integer maps and the full NeurID decoding source are included. The patched NeurID exporter affects future replay output only; it does not manufacture identities for old CSV rows.
- StatAtlas saves Top-1 and Hungarian identities, but not the ordered Top-5 identity list.
- The normalized CRF_ID evaluator output saves Top-1 identity and correctness fields. Original MATLAB output matrices and sidecars are now included for complete independent decoding.
- Blank identity fields mean “not present in the saved artifact,” not an inferred unknown prediction.
- For CPD/fDNC, exact canonical Hungarian accuracy remains reproducible from each fold metrics JSON because their native query rows are a proven subset of canonical queries; however, the identity of each correctly assigned query cannot be recovered from the compact saved CSV.
