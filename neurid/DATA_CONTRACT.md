# NPZ data contract

Each file describes one observed neuronal population.

| Key | Shape | Required | Meaning |
|---|---:|---:|---|
| `activity_raw` | `[N, T]` | yes | calcium activity trace for every observed neuron |
| `xyz` | `[N, 3]` | yes | three-dimensional neuron coordinates |
| `cell_id` | `[N]` | yes | identity label used only for supervision/evaluation |
| `labeled_mask` | `[N]` | no | whether an identity annotation is present |
| `certain_mask` | `[N]` | no | whether the annotation passes source confidence criteria |
| `clean_mask` | `[N]` | no | dataset-specific quality mask |
| `valid_xyz_mask` | `[N]` | no | finite/usable-coordinate mask |
| `recording_uid` | scalar | no | stable recording identifier |

`N` may differ between recordings and `T` need not match across recordings.
Activity is resampled independently inside each recording; no cross-recording
time alignment is required.

All finite-coordinate neurons are retained as population context. Supervision
uses valid, nonempty identities that occur once in a recording and satisfy all
available label-quality masks. Duplicate or unknown identities are not
resolved heuristically.

Prepared roots included with this repository:

- `data/atanas/fold_0` through `fold_4`;
- `data/kato_rld/fold_0` through `fold_4`;
- `data/zebrafish/fold_1` through `fold_8`.

Every fold contains `train/`, `val/`, and `test/`. Worm manifests additionally
map recording filenames to biological grouping keys used by the
individual-balanced atlas.
