# Table 1 / Table 2 baseline-core audit

Table 2 is keyed to the Table 1 method implementation. Dataset-specific
reference construction may differ, but the model, scoring rule and assignment
mechanism must not be replaced by a different algorithm under the same name.

| Table 2 row | Audit decision | Result action |
|---|---|---|
| StatAtlas | Former zebrafish code used pair-local median/RMS normalization, pair ICP, one shared residual covariance and raw Mahalanobis scores. This is not the Table 1 `train_official_atlas` → `label_free_align` → `atlas_identity_posterior` → posterior-overlap pipeline. | Rebuilt and reran all eight `select → test → aggregate` folds. |
| CRF_ID† | Former zebrafish code used pycpd unary scores plus a custom iterative Hungarian relation update. This is not the Table 1 fully connected angle CRF with uniform node potentials, UGM-LBP and iterative clamping. | Replaced by the byte-identical Table 1 MATLAB adapter and reran all eight folds. |
| CPD, fDNC, NuCLR, GeoTransformer, NGM-v2, Vanilla FGW, FUGW and Ours | No same-name algorithm substitution was introduced by this correction; their q/r score construction, candidate population and held-out test records are unchanged. | Retained. The shared extra-reference manifest is exposed to every row; these pairwise cores do not consume atlas-reference records. |

For each of the original 101 physical q/r pairs, the two updated atlas methods
use the identical reference list in `table1_core_reference_manifest.csv`: q and
r are excluded, and all remaining unique same-fish timepoints are aligned by
stable `original_row`. This changes the original single-reference input
condition and is stated explicitly in the table documentation and
`REFERENCE_CONFIGS.json`.

StatAtlas uses `score_fn = P_q @ P_r.T`; CRF_ID† uses
`score_fn = B_q @ B_r.T`. Both pass the resulting q/r matrix to the shared
ranking and Hungarian implementation. Endpoint tracking IDs are absent from
atlas construction and MATLAB/Python inference inputs; they are opened only by
the metric writer after the complete score matrix exists.

The former pairwise Python adaptations remain in
`statatlas_crfid_pairwise_lofo8_v1/` for provenance and are not Table 2 rows.
