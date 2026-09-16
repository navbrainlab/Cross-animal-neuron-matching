# NeuRID package

`mprt_net.model.NeuRID` is the sole public model. It implements the manuscript
pipeline from per-recording geometry/activity encoding through population
relations, contextualization, fixed-atlas construction, and differentiable
partial matching.

The package name remains `mprt_net` for checkpoint compatibility. Public class
names are `NeuRID` (an alias of `MPRTNet`), `ModelConfig`, `PopulationEncoding`,
and `MPRTOutput`.

See `DATA_CONTRACT.md` for NPZ fields and `../MODEL_CODE_GUIDE.md` for the
module map.
