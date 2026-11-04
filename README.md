# anofox_ml — machine learning in DuckDB

`anofox_ml` brings the [anofox-ml](https://github.com/sipemu/anofox-ml) Rust
library (scikit-learn-compatible estimators, checked against scikit-learn)
into SQL: fit a model per group with an aggregate, store it as a BLOB, and
predict with it in any later query.

> **Status: early development.** The first functions below work; the rest of
> the design (predict_proba, transform, clustering, outlier scores, metrics,
> fit-predict macros) is tracked in sipemu/anofox-ml#91.

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
| `ml_predict(model, [x1, x2, …])` | Predict with a model BLOB |
| `ml_model_kind(model)` | The estimator kind stored in a model BLOB |
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

-- what is available, and with which defaults
SELECT kind, default_params FROM ml_estimators();
```

Models are stored in anofox-ml's versioned binary format (MessagePack with
field names, zstd-compressed), which stays readable as the library evolves.
Rows with a NULL target or feature list are skipped when fitting. Each fit runs
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
