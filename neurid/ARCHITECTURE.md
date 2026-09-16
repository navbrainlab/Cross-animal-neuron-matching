# NeuRID architecture

For every recording, NeuRID normalizes 3-D coordinates and encodes coordinates
and activity traces separately. Directed pair descriptors combine relative
displacement/distance with trace and derivative correlations. A
relation-conditioned Transformer uses those descriptors in both attention
weights and messages.

The final relation vector combines the pair descriptor with the difference and
element-wise product of contextualized node embeddings. Node cosine similarity
initializes a partial transport plan with a shared unmatched state. Two
differentiable refinement steps compare directed relations under the current
soft assignment and rerun log-domain Sinkhorn normalization.

Training uses bidirectional focal supervision on known matches and synthetic
unmatched examples. Validation selects the checkpoint. The fixed inference
atlas averages training-recording encodings by individual and then across
individuals, preventing animals with more recordings from dominating.

The implementation is in `mprt_net/model.py`; training and atlas construction
are in `mprt_net/train.py`.
