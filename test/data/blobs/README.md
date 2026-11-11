# Model BLOB fixtures

Models saved by older versions of the extension. `test/sql/ml_blob_compat.test` checks that the current build still loads them and reproduces the predictions they gave when they were saved.

- `models_<version>.csv`: `case, kind, method, blob_hex`, where `blob_hex` is the model BLOB as written by `ml_fit_agg`.
- `expected_<version>.csv`: `case, row, x, expected`, the output of `method` on four test rows, computed by that version.

`v0.2` was written by the extension at `99f00f0` (anofox-ml 0.2.0), `v0.4` at `fb869aa` (anofox-ml 0.4.0, with PerpetualBooster; it adds CatBoost, split conformal and LedoitWolf cases). The training data is deterministic: 60 rows, `x = [i / 10, (7i mod 11) / 3]`, and `y = 2·x1 − x2 + (13i mod 5) / 10`. Classification targets are `i mod 3` (as the labels `hi` / `lo` / `mid`) and `y > 2` (for GaussianNB).

`v0.5` was written by `scripts/blob_fixture.sql` (anofox-ml 0.5.0, extension at f0fc3d8). It adds LightGBM, CatBoost classification, voting, a Gaussian process, the `VARCHAR` encoders (`x` is a `VARCHAR[]`) and a text vectorizer (`x` is one document).

When anofox-ml is bumped, write a fixture for the version being replaced with `build/release/duckdb < scripts/blob_fixture.sql` (run in this directory) before bumping, and rename its output; don't regenerate the existing ones.
