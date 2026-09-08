# Baseline methods

NeuRID is the primary model and lives in `../neurid/`. Everything in this
directory is comparison code, an official-method adapter, or a compatibility
overlay.

| Directory | Contents |
|---|---|
| `official/` | Locked protocols and project adapters for official baseline repositories |
| `adapters/` | GeoTransformer, RGM, and ThinkMatch/NGM-v2 overlays |
| `fdnc/` | Standalone fDNC evaluation entry point |
| `gwot_md/` | Audited GWOT-MD paper-protocol implementation |
| `stat_atlas/` | Statistical Atlas evaluation entry point |
| `atanas_locked.py` | Shared locked-Atanas baseline utilities |

Upstream third-party repositories and weights are not vendored. Frozen URLs
and revisions are recorded in `../docs/THIRD_PARTY.md` and
`official/manifests/official_repo_revisions.tsv`.
