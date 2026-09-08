# Dataset preparation workflows

This directory is restricted to downloading source datasets, extracting the
unified NPZ representation, and creating leakage-safe worm-level splits.
NeuRID launchers live in `../scripts/neurid/`; baseline launchers live in
`../scripts/benchmarks/` and `../baselines/`.

| Directory | Retained responsibility |
|---|---|
| `atanas/` | DANDI extraction and date-disjoint split preparation |
| `kk_000692/` | KK aligned-data preparation |
| `dunn_001623/` | Official Dunn/RLD download and split preparation |
| `sk1_000565/` | SK1 NWB-to-NPZ preparation |
| `copper_boundary/` | Copper boundary split preparation |
| `dag_nwabudike_kang/` | Dag/WormWideWeb split preparation |
| `cross_domain/` | Cross-domain protocol-list preparation |

The protocol-lock files retain some historical launcher paths as immutable
provenance. Those paths are not active entry points in this source release.
