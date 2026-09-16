# Official baseline adapters

This directory contains only project-specific adapters for official baseline
implementations used in the manuscript: fDNC, NuCLR, and StatAtlas. NeuRID is
implemented under `../../neurid/`; GeoTransformer and NGM-v2 overlays are in
`../adapters/`.

Upstream repositories are not vendored. Clone the revisions recorded in
`manifests/official_repo_revisions.tsv` separately, keep their source trees
unchanged, and run the corresponding adapter:

- `adapters/fdnc_finetune/`
- `adapters/nuclr_official_scratch50k/`
- `adapters/stat_atlas/`

The adapters follow the manuscript protocol: animal-level held-out folds,
training-only fitting, validation-only selection, and test-only reporting.
Method-specific environments are recommended because upstream dependency
versions differ.
