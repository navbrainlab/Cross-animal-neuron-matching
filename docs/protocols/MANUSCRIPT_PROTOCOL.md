# Manuscript model protocol

The only public NeuRID model is `neurid/mprt_net/model.py`. The worm evaluation
uses five locked folds and seed 42. Checkpoints are selected on each fold's
validation split, the fixed atlas is built from outer-training recordings, and
the test split is evaluated only after selection.

Atanas and Kato/RLD use canonical identity slots. Zebrafish identities are
pair-local, so its eight held-out-fish folds form the statistical units.

The release intentionally does not expose switches or launchers for historical
model versions as alternative primary methods.
