# Public release scope

The public tree contains the manuscript-synchronized NeuRID implementation,
the three prepared manuscript datasets, every frozen manuscript result,
corresponding analysis/evaluation code, baseline adapters, and data-preparation
provenance.

Excluded from this tree:

- post-paper proposed-revision, dynamic/residual-atlas, and transfer variants;
- pairwise-initialized, full-support, hard-negative, and global-unary candidate runs;
- historical CV5 x 3-seed development artifacts;
- checkpoints, logs, caches, and generated run directories;
- datasets not named in the manuscript.

Those materials are retained locally outside the Git repository under
`archive/neurid_github_pre_public_cleanup_20260916/`.
