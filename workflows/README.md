# Dataset preparation workflows

This directory is restricted to downloading source datasets, extracting the
unified NPZ representation, and creating leakage-safe worm-level splits.
Production model launchers live in `../engines/`.

| Directory | Retained responsibility |
|---|---|
| `atanas/` | DANDI extraction and date-disjoint split preparation |
| `kk_000692/` | KK aligned-data preparation |
| `dunn_001623/` | Official Dunn/RLD download and split preparation |
| `sk1_000565/` | SK1 NWB-to-NPZ preparation |
| `copper_boundary/` | Copper boundary split preparation |
| `dag_nwabudike_kang/` | Dag/WormWideWeb split preparation |
| `cross_domain/` | Cross-domain protocol-list preparation |

Historical training, baseline, GAT, zero-shot, and result-summary workflows were
moved to `../archive/legacy_workflows/`. They are not active entry points.
