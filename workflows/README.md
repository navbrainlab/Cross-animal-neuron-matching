# Dataset preparation workflows

These scripts document how the included prepared data were obtained from
their public sources.

| Path | Responsibility |
|---|---|
| `atanas/` | extract activity, coordinates, and identities from DANDI 000776 |
| `atanas_multifold_v1/` | immutable earlier split provenance retained for audit only |
| `dunn_001623/` | download Zenodo 17353307, convert records to NPZ, and prepare grouped folds |

Ready-to-run prepared folds live in `../data/`; users do not need to execute
these workflows for the bundled experiments. Some immutable provenance files
retain their original source-machine paths and are not runtime configuration.
