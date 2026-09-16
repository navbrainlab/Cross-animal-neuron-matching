# NeuRID model code map

The public package is the checkpoint-compatible implementation that produced
the manuscript's formal results.

| File | Responsibility |
|---|---|
| `neurid/mprt_net/config.py` | `ModelConfig` and the locked architecture/matcher controls |
| `neurid/mprt_net/model.py` | multimodal encoders, directed relations, population encoder, atlas state, and iterative matching |
| `neurid/mprt_net/layers.py` | relation-conditioned attention blocks |
| `neurid/mprt_net/relations.py` | geometry/activity features and relation fusion |
| `neurid/mprt_net/sinkhorn.py` | log-domain partial Sinkhorn and relation cost |
| `neurid/mprt_net/losses.py` | bidirectional focal matching objectives |
| `neurid/mprt_net/data.py` | NPZ loading, activity resampling, and supervised targets |
| `neurid/mprt_net/train.py` | training and validation checkpoint selection |
| `neurid/mprt_net/build_anchored_atlas.py` | train-only population-atlas construction |
| `neurid/mprt_net/evaluate.py` | held-out evaluation and query export |

Public API:

```python
from mprt_net import ModelConfig, NeuRID

model = NeuRID(ModelConfig())
query = model.encode_population(query_sample)
reference = model.encode_population(reference_sample)
output = model.match_encodings(query, reference)
```

Alternative and post-paper candidate implementations remain only in the local,
non-Git archive.
