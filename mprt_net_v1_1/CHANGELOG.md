# v0.1.1

This is an evaluation-protocol correction, not an architecture revision.

Changed:

- Benchmark ranking excludes the dustbin and is named explicitly as
  `top1_real`, `top5_real`, and `mrr_real`.
- Dustbin-inclusive metrics and the dustbin top-1 rate remain available as
  separate diagnostics.
- Training selects `best.pt` using validation `top1_real`.
- Evaluation can save one JSON record per animal pair.
- `mprt_net.compare` performs paired query analysis, rescue/harm counting,
  pair wins/ties/losses, pair-cluster bootstrap, and animal-resampled
  bootstrap.
- New launchers run the four Atanas seed-42 factorial variants in fresh output
  directories and compare them after training.
- Training refuses to overwrite an existing run unless explicitly permitted.

Unchanged byte-for-byte from v0.1:

- `model.py`
- `relations.py`
- `layers.py`
- `sinkhorn.py`
- `losses.py`
- `data.py`
- `config.py`
- `self_check.py`

Therefore node/activity encoding, fusion, population attention, relation-field
construction, relational transport, augmented Sinkhorn, focal objective, and
default `transport_steps=2` are identical to the pilot.
