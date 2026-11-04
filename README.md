<p align="center">
  <img src="docs/assets/logo.svg" alt="Anofox Logo" width="80" height="80">
</p>

<h1 align="center">Anofox ML</h1>
<p align="center"><strong>Machine Learning for DuckDB</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSL%201.1-blue.svg" alt="License: BSL 1.1"></a>
  <a href="https://duckdb.org"><img src="https://img.shields.io/badge/DuckDB-1.5.6-green.svg" alt="DuckDB"></a>
  <a href="https://github.com/DataZooDE/anofox-ml/actions/workflows/MainDistributionPipeline.yml"><img src="https://github.com/DataZooDE/anofox-ml/actions/workflows/MainDistributionPipeline.yml/badge.svg?branch=main" alt="Build"></a>
  <img src="https://img.shields.io/badge/estimators-104-brightgreen.svg" alt="104 estimators">
  <img src="https://img.shields.io/badge/status-early%20development-orange.svg" alt="Status: early development">
</p>

A DuckDB extension that brings the [anofox-ml](https://github.com/sipemu/anofox-ml) Rust library into SQL: 104 scikit-learn-compatible estimators, checked against scikit-learn. You fit a model per group with an aggregate, store it as a `BLOB` and predict with it in any later query.

> **Status: early development.** The functions below work and are tested. Metrics, fit-predict table macros and tuning are tracked in the [issues](https://github.com/DataZooDE/anofox-ml/issues) and in sipemu/anofox-ml#91.


## ✨ Key Features

### 🌲 Supervised Learning
- **Trees & Ensembles**: Decision trees, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting, LightGBM-style boosting, AdaBoost, Bagging
- **Kernel & Distance Methods**: SVC/SVR, Nu-SVC/SVR, linear SVMs, kNN and radius neighbours, kernel ridge
- **Probabilistic**: Gaussian, Bernoulli, multinomial, complement and categorical naive Bayes; LDA / QDA
- **Neural Networks**: MLP classifier and regressor
- **Linear Models**: OLS, Ridge, Lasso, Elastic Net (+ CV variants), LARS, OMP, Bayesian Ridge, ARD, Huber, Quantile, RANSAC, Theil-Sen, logistic, Poisson/Gamma/Tweedie GLMs, SGD, Passive-Aggressive

### 🔍 Unsupervised Learning
- **Clustering**: KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, OPTICS, MeanShift, Birch, Affinity Propagation, Agglomerative, Spectral, Gaussian Mixture, Bayesian Gaussian Mixture
- **Outlier Detection**: Isolation Forest, Local Outlier Factor, One-Class SVM
- **Decomposition & Manifold**: PCA, Incremental PCA, Kernel PCA, Truncated SVD, FastICA, NMF, MDS, Isomap, LLE, t-SNE

### 🧰 Preprocessing
- **Scaling**: Standard, MinMax, MaxAbs and Robust scalers, Normalizer, Power and Quantile transformers
- **Feature Construction**: polynomial features, B-splines, restricted cubic splines, binning, binarizer, target encoding
- **Imputation & Selection**: simple, kNN and iterative imputers, missing indicator, variance threshold, SelectKBest, mutual information

### 💾 Models as Data
- **Stored as `BLOB`s**: a fitted model is an ordinary column value. Keep it in a table, join it to new rows, write it to Parquet.
- **Versioned format**: MessagePack with field names, zstd-compressed, so stored models stay readable as the library evolves
- **Introspection**: `ml_model_info` returns kind, feature count, classes, supported methods and parameters

### ⚡ Performance
- **Parallel across groups**: DuckDB fits one model per `GROUP BY` group on all cores
- **Native Rust core**: no Python runtime, no data copies out of the database
- **Cached decoding**: a model is decoded once per thread, not once per predicted row

### 🎨 User-Friendly API
- **Generic**: one set of functions for every estimator, selected by `kind`
- **JSON parameters**: merged into the estimator defaults, with an error listing the valid names when you misspell one
- **Multi-language**: use it from Python, R, Julia, Node.js, Go, Java, Rust or C++ through DuckDB's bindings

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Installation](#installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Estimators](#-estimators)
- [Behaviour Notes](#-behaviour-notes)
- [Multi-Language Support](#-multi-language-support)
- [Development](#-development)
- [License](#-license)


## Installation

The extension is not published as a DuckDB community extension yet; that is on hold until the API has matured. Build it from source for now.

### From Source

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-ml.git
cd anofox-ml

GEN=ninja make release       # or: make release -j$(nproc)

# The extension is built to:
# build/release/extension/anofox_ml/anofox_ml.duckdb_extension
# and build/release/duckdb starts a shell with it already loaded.
```

The `duckdb` submodule is pinned to v1.5.6, so a source build targets DuckDB v1.5.6.

## 🚀 Quick Start

This example is self-contained: it generates synthetic store sales data, fits one gradient-boosting model per store, evaluates it on a hold-out set, then clusters the stores and scores outliers.

```sql
LOAD anofox_ml;

-- 3 stores, units depend on weekday, price and promotion
CREATE OR REPLACE TABLE sales AS
SELECT *, 100 + 10 * dow - 20 * price + 30 * promo + 5 * noise AS units
FROM (
    SELECT i AS id,
           'store_' || (i % 3) AS store,
           i % 7 AS dow,
           0.5 + (hash(i, 'p') % 1000) / 1000 AS price,
           (hash(i, 'q') % 5 = 0)::INT AS promo,
           (hash(i, 'n') % 1000) / 1000 AS noise
    FROM range(3000) t(i)
);
CREATE OR REPLACE TABLE train AS SELECT * FROM sales WHERE id < 2400;
CREATE OR REPLACE TABLE test  AS SELECT * FROM sales WHERE id >= 2400;

-- one gradient-boosting model per store, stored as a BLOB
CREATE OR REPLACE TABLE models AS
SELECT store,
       ml_fit_agg('HistGradientBoostingRegressor', units, [price, promo, dow],
                  '{"n_estimators": 200, "learning_rate": 0.05}') AS model
FROM train
GROUP BY store;

-- predict with the stored models and compute the RMSE per store
SELECT store,
       sqrt(avg(power(units - ml_predict(m.model, [price, promo, dow]), 2))) AS rmse
FROM test t JOIN models m USING (store)
GROUP BY store ORDER BY store;
-- store_0 ≈ 1.59, store_1 ≈ 1.70, store_2 ≈ 1.84

-- what is stored in a model
SELECT store, ml_model_info(model).kind, ml_model_info(model).n_features FROM models;

-- per-row cluster labels: ORDER BY inside both aggregates keeps ids and labels aligned
SELECT store,
       unnest(list(id ORDER BY id)) AS id,
       unnest(ml_cluster_agg('KMeans', [price, dow], '{"n_clusters": 3}' ORDER BY id)) AS cluster
FROM train GROUP BY store;

-- the most anomalous rows (lower score = more anomalous, as in scikit-learn)
SELECT unnest(list(id ORDER BY id)) AS id,
       unnest(ml_outlier_score_agg('IsolationForest', [price, units] ORDER BY id)) AS score
FROM train ORDER BY score LIMIT 3;
```

More patterns:

```sql
-- class probabilities, in the order of the stored classes
SELECT ml_model_info(model).classes, ml_predict_proba(model, [x1, x2])
FROM churn_model, customers;

-- unsupervised fit: standardise features, then transform
CREATE TABLE scaler AS SELECT ml_fit_agg('StandardScaler', [price, promo]) AS model FROM train;
SELECT ml_transform(model, [price, promo]) FROM train, scaler;

-- every estimator and its defaults
SELECT kind, default_params FROM ml_estimators();
```

## 📚 API Reference

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
| `ml_cluster_agg(kind, [x1, x2, …] [, params])` | Aggregate: fit a clustering (or outlier) estimator on the group and return one label per row (`BIGINT[]`; `-1` is noise for DBSCAN-like methods) |
| `ml_outlier_score_agg(kind, [x1, x2, …] [, params])` | Aggregate: fit an outlier detector on the group and return one score per row (`DOUBLE[]`; lower is more anomalous, as in scikit-learn) |
| `ml_model_info(model)` | `STRUCT` of kind, `n_features`, `classes`, `capabilities` (supported methods), `params` (JSON), `format_version` and `crate_version` |
| `ml_model_kind(model)` | The estimator kind stored in a model BLOB (without decoding the model) |
| `ml_version()` | Version of the bundled anofox-ml FFI layer |

### Roadmap

| Topic | Issue |
|---|---|
| Metric aggregates (`ml_r2`, `ml_rmse`, `ml_roc_auc`, `ml_log_loss`, …) | [#1](https://github.com/DataZooDE/anofox-ml/issues/1) |
| Fit-predict table macros (`ml_fit_predict_by`, `ml_cluster_by`, …) | [#2](https://github.com/DataZooDE/anofox-ml/issues/2) |
| Sample weights and a memory guard for large groups | [#3](https://github.com/DataZooDE/anofox-ml/issues/3) |
| Hyperparameter tuning (`ml_tune_agg`, rustuna) | [#5](https://github.com/DataZooDE/anofox-ml/issues/5) |
| Conformal prediction intervals | [#6](https://github.com/DataZooDE/anofox-ml/issues/6) |
| Parallelism within a single fit | [#7](https://github.com/DataZooDE/anofox-ml/issues/7) |

## 🧮 Estimators

Pass the `kind` exactly as listed. `SELECT * FROM ml_estimators()` shows the defaults for each one.

| Category | Kinds |
|---|---|
| **Trees & Ensembles** | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `LgbmClassifier`, `LgbmRegressor`, `AdaBoostClassifier`, `AdaBoostRegressor`, `BaggingClassifier`, `BaggingRegressor` |
| **SVM** | `Svc`, `Svr`, `NuSvc`, `NuSvr`, `LinearSvc`, `LinearSvr` |
| **Neighbours** | `KnnClassifier`, `KnnRegressor`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor` |
| **Naive Bayes** | `GaussianNB`, `BernoulliNB`, `MultinomialNB`, `ComplementNB`, `CategoricalNB` |
| **Discriminant Analysis** | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |
| **Neural Networks** | `MlpClassifier`, `MlpRegressor` |
| **Linear & GLM** | `OlsRegressor`, `RidgeRegressor`, `RidgeCrossValidated`, `LassoRegressor`, `LassoCrossValidated`, `ElasticNetRegressor`, `ElasticNetCrossValidated`, `Lars`, `OrthogonalMatchingPursuit`, `BayesianRidge`, `ARDRegression`, `HuberRegressor`, `QuantileRegressor`, `RansacRegressor`, `TheilSenRegressor`, `KernelRidge`, `LogisticRegressor`, `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`, `SgdClassifier`, `SgdRegressor`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor` |
| **Clustering** | `KMeans`, `MiniBatchKMeans`, `Dbscan`, `Hdbscan`, `Optics`, `MeanShift`, `Birch`, `AffinityPropagation`, `AgglomerativeClustering`, `SpectralClustering`, `GaussianMixture`, `BayesianGaussianMixture` |
| **Outlier Detection** | `IsolationForest`, `LocalOutlierFactor`, `OneClassSvm` |
| **Decomposition & Manifold** | `Pca`, `IncrementalPca`, `KernelPca`, `TruncatedSvd`, `FastIca`, `Nmf`, `ClassicalMds`, `Isomap`, `LocallyLinearEmbedding`, `TSne` |
| **Preprocessing** | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `Binarizer`, `PowerTransformer`, `QuantileTransformer`, `KBinsDiscretizer`, `PolynomialFeatures`, `SplineTransformer`, `RcsTransformer`, `TargetEncoder`, `SimpleImputer`, `KnnImputer`, `IterativeImputer`, `MissingIndicator`, `VarianceThreshold`, `SelectKBest`, `MutualInformationSelector` |

Linear models and GLMs **with statistical inference** (standard errors, tests, diagnostics) live in [anofox-statistics](https://github.com/DataZooDE/anofox-statistics). The linear models here are for prediction.

## 📝 Behaviour Notes

- **NULLs:** rows with a NULL target or feature list are skipped when fitting. `ml_cluster_agg` and `ml_outlier_score_agg` return a NULL entry for such rows.
- **Row order:** `ml_cluster_agg` and `ml_outlier_score_agg` return one entry per input row in input order. That order is only well defined with `ORDER BY` inside the aggregate.
- **Classes:** for models with `predict_proba`, the sorted distinct training targets are stored as `classes`.
- **Feature count:** predicting with a feature list of the wrong length is an error.
- **Parallelism:** DuckDB parallelises across groups. Each fit currently runs single-threaded inside a DuckDB worker, so an ungrouped fit uses one core. [#7](https://github.com/DataZooDE/anofox-ml/issues/7) proposes letting a single fit use the idle cores.
- **Determinism:** estimators with randomness take a `seed` parameter with a fixed default, so the same query gives the same model.

## 🌍 Multi-Language Support

**Write SQL once, use everywhere!** The extension works from any language with DuckDB bindings.

| Language | Status | Notes |
|----------|--------|-------|
| **Python** | ✅ | Via DuckDB Python bindings |
| **R** | ✅ | Via DuckDB R bindings |
| **Julia** | ✅ | Via DuckDB Julia bindings |
| **C++** | ✅ | Via DuckDB C++ bindings |
| **Rust** | ✅ | Via DuckDB Rust bindings |
| **Node.js** | ✅ | Via DuckDB Node bindings |
| **Go** | ✅ | Via DuckDB Go bindings |
| **Java** | ✅ | Via DuckDB JDBC driver |

---

## 📦 Development

### Prerequisites

- C++17 compiler, CMake ≥ 3.20, Make or Ninja
- Rust toolchain (stable)

### Build and Test

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-ml.git
cd anofox-ml

GEN=ninja make release            # build/release/duckdb with the extension loaded
make test                         # SQL logic tests in test/sql
cargo test                        # Rust FFI tests
cargo fmt --all                   # format
cargo clippy --all-targets        # lint
```

### Layout

| Path | Contents |
|---|---|
| `src/` | C++ extension: thin DuckDB wrappers around the C ABI |
| `crates/anofox-ml-ffi/` | Rust `staticlib`: generic C ABI over [`anofox-ml-registry`](https://github.com/sipemu/anofox-ml/tree/master/crates/anofox-ml-registry) |
| `test/sql/` | sqllogictests |

The estimators come from [anofox-ml](https://github.com/sipemu/anofox-ml). The registry crate is consumed from its `master` branch until it is published on crates.io.

### Verify Installation

```bash
./build/release/duckdb -c "SELECT ml_version(), count(*) AS estimators FROM ml_estimators();"
```

## 📄 License

**Business Source License 1.1 (BSL 1.1)**, like the other DataZoo DuckDB extensions.

### Key Points

✅ **Free for production use** - Use internally in your business  
✅ **Free for development** - Build applications with it  
✅ **Free for research** - Academic and research use  

❌ **Cannot offer as hosted service** - No SaaS offerings to third parties  
❌ **Cannot embed in commercial product** - For third-party distribution  

🔄 **Converts to MPL 2.0** - After 5 years from first publication

See [LICENSE](LICENSE) for full terms. The bundled [anofox-ml](https://github.com/sipemu/anofox-ml) library is licensed under MIT OR Apache-2.0.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DataZooDE/anofox-ml/issues)
- **Email**: sm@data-zoo.de

## 🎓 Citation

If you use this extension in research, please cite:

```bibtex
@software{anofox_ml,
  title = {Anofox ML: Machine Learning for DuckDB},
  author = {Joachim Rosskopf, Simon Müller, DataZoo GmbH},
  year = {2026},
  url = {https://github.com/DataZooDE/anofox-ml}
}
```

## 🏆 Acknowledgments

Built on top of:
- [DuckDB](https://duckdb.org) - Amazing analytical database
- [anofox-ml](https://github.com/sipemu/anofox-ml) - scikit-learn-compatible machine learning in Rust

Special thanks to the DuckDB team for making extensions possible!

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

---

**Made with ❤️ by the Anofox Team**

⭐ **Star us on GitHub** if you find this useful!

📢 **Follow us** for updates: [@datazoo](https://www.linkedin.com/company/datazoo/)
