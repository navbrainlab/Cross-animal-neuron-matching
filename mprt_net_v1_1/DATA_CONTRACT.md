# Verified data contract

The implementation was written against the uploaded contract report and one
real NPZ sample from each dataset.

| Dataset | Root | Direct split counts | Typical node/time shape |
| --- | --- | --- | --- |
| Atanas | `Data/Atanas_SF_unified_000776/date_disjoint_v1/full` | 27 train / 6 val / 5 test | about 109--144 nodes, 1600--1615 frames |
| RLD | `Data/Dunn_001623/date_disjoint_full95_v1` | 67 train / 12 val / 16 test | about 71--106 nodes in sampled files, 1000--1500 frames |

Required arrays:

| Key | Shape | Use |
| --- | --- | --- |
| `activity_raw` | `[N,T]` | Node dynamics and within-animal functional relations |
| `xyz` | `[N,3]` | Node geometry and multi-scale soft spatial relations |
| `cell_id` | `[N]` | Training/evaluation targets only; never an input feature |

Supported masks are `labeled_mask`, `certain_mask`, `clean_mask`, and
`valid_xyz_mask`. All finite-coordinate neurons are retained as context.
Supervision uses the intersection of the first three label masks and a valid,
nonempty identity. Repeated identities inside one recording are excluded from
pair targets rather than resolved heuristically.

Additional metadata such as `recording_uid`, `timestamps`, sampling rate, ROI
indices, and dataset provenance is accepted but not required by the model.

The RLD files in the uploaded report include `all/` and
`supervised_nonempty/` copies in addition to the direct `train/val/test`
directories. `PairIndex` reads only `<root>/<requested split>/*.npz`, so those
copies are not double-counted.

No stimulus interval metadata appeared in the uploaded RLD arrays. The loader
therefore makes no stimulus assumptions.
