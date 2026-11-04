# anofox_ml — machine learning in DuckDB

`anofox_ml` brings the [anofox-ml](https://github.com/sipemu/anofox-ml) Rust
library (scikit-learn-compatible estimators, checked against scikit-learn)
into SQL: fit a model per group with an aggregate, store it as a BLOB, and
predict with it in any later query.

> **Status: early development.** The functions below work; the rest of the
> design (clustering and outlier aggregates, metrics, fit-predict macros) is
> tracked in sipemu/anofox-ml#91.

Linear models and GLMs with statistical inference live in
[anofox-statistics](https://github.com/DataZooDE/anofox-statistics); this
extension focuses on machine-learning estimators (trees and ensembles, SVM,
kNN, naive Bayes, discriminant analysis, MLP, clustering, outlier detection,
decomposition and preprocessing).

## Functions

| Function | Description |
|---|---|
| `ml_estimators()` | Table of every estimator `kind` with its default parameters (JSON) |
| `ml_fit_agg(kind, y, [x1, x2, …] [, params])` | Aggregate: fit a model per group, returns a model `BLOB`. `params` is a JSON object merged into the defaults |
| `ml_fit_agg(kind, [x1, x2, …] [, params])` | The same without a target, for unsupervised estimators (scalers, PCA, clustering, …) |
| `ml_predict(model, [x1, x2, …])` | Predict with a model BLOB (`DOUBLE`) |
| `ml_predict_proba(model, [x1, x2, …])` | Class probabilities (`DOUBLE[]`), in the order of `ml_model_info(model).classes` |
| `ml_decision_function(model, [x1, x2, …])` | Decision function values (`DOUBLE[]`) |
| `ml_transform(model, [x1, x2, …])` / `ml_inverse_transform(…)` | Transformed features (`DOUBLE[]`) |
| `ml_score_samples(model, [x1, x2, …])` | Per-sample scores, e.g. outlier scores or log-likelihoods (`DOUBLE[]`) |
| `ml_model_info(model)` | `STRUCT` of kind, `n_features`, `classes`, `capabilities` (supported methods), `params` (JSON), `format_version` and `crate_version` |
| `ml_model_kind(model)` | The estimator kind stored in a model BLOB (without decoding the model) |
| `ml_version()` | Version of the bundled anofox-ml FFI layer |

```sql
LOAD anofox_ml;

-- one gradient-boosting model per store
CREATE TABLE models AS
SELECT store,
       ml_fit_agg('HistGradientBoostingRegressor', sales, [price, promo, dow],
                  '{"max_iter": 200, "learning_rate": 0.05}') AS model
FROM train
GROUP BY store;

-- predict with the stored models
SELECT t.*, ml_predict(m.model, [t.price, t.promo, t.dow]) AS forecast
FROM test t JOIN models m USING (store);

-- class probabilities, columns in the order of the stored classes
SELECT ml_model_info(model).classes, ml_predict_proba(model, [x1, x2]) FROM churn_model, customers;

-- unsupervised: standardise features
CREATE TABLE scaler AS SELECT ml_fit_agg('StandardScaler', [price, promo]) AS model FROM train;
SELECT ml_transform(model, [price, promo]) FROM train, scaler;

-- what is available, and with which defaults
SELECT kind, default_params FROM ml_estimators();
```

Models are stored in anofox-ml's versioned binary format (MessagePack with
field names, zstd-compressed), which stays readable as the library evolves.
Rows with a NULL target or feature list are skipped when fitting. For models
with `predict_proba`, the sorted distinct training targets are stored as
`classes`; predicting with a feature list of the wrong length is an error. Each fit runs
single-threaded inside a DuckDB worker; DuckDB parallelises across groups.

## Building

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-ml.git
cd anofox-ml
GEN=ninja make release      # build/release/duckdb with the extension loaded
make test                   # SQL logic tests in test/sql
cargo test                  # Rust FFI tests
```

Requires a C++17 compiler, CMake ≥ 3.20 and a Rust toolchain.

## License

Business Source License 1.1 (see [LICENSE](LICENSE)), like the other
DataZoo DuckDB extensions. The bundled
[anofox-ml](https://github.com/sipemu/anofox-ml) library is licensed under
MIT OR Apache-2.0.

## Telemetry

This extension collects anonymous usage telemetry to help improve the product. Telemetry is **enabled by default** and includes:

- Extension load events (extension name, version, platform)
- Function execution events (which functions are used)
- No personal data or query contents are collected

### Disabling Telemetry

**Environment Variable:**
```bash
export DATAZOO_DISABLE_TELEMETRY=1
```

**DuckDB SQL Setting:**
```sql
SET anofox_telemetry_enabled = false;
```

For more information, see the [posthog-telemetry](https://github.com/DataZooDE/posthog-telemetry) repository.
