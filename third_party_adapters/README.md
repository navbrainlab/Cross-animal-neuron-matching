# Third-party adapters

These files are project-specific overlays for frozen upstream checkouts. They
contain only the custom experiment code; upstream repositories, datasets,
weights, logs, and build products are not vendored.

- `geotransformer/`: copy the experiment directories and launchers into the
  root of the frozen GeoTransformer checkout recorded in `docs/THIRD_PARTY.md`.
- `rgm/`: copy the three launchers into the frozen RGM checkout root.
- `thinkmatch/python310-ortools.patch`: apply to the frozen ThinkMatch checkout
  before running NGM-v2 under the Python 3.10 environment used here.

Example:

```bash
git -C /path/to/ThinkMatch apply --unidiff-zero \
  /path/to/this/repo/third_party_adapters/thinkmatch/python310-ortools.patch
```
