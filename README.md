<p align="center">
  <img src="docs/assets/logo.svg" alt="Anofox Logo" width="80" height="80">
</p>

<h1 align="center">Anofox ML</h1>
<p align="center"><strong>Machine Learning for DuckDB</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSL%201.1-blue.svg" alt="License: BSL 1.1"></a>
  <a href="https://duckdb.org"><img src="https://img.shields.io/badge/DuckDB-1.5.6-green.svg" alt="DuckDB"></a>
  <a href="https://github.com/DataZooDE/anofox-ml/actions/workflows/MainDistributionPipeline.yml"><img src="https://github.com/DataZooDE/anofox-ml/actions/workflows/MainDistributionPipeline.yml/badge.svg?branch=main" alt="Build"></a>
  <img src="https://img.shields.io/badge/estimators-196-brightgreen.svg" alt="196 estimators">
  <a href="#-claude-code-skills"><img src="https://img.shields.io/badge/Claude_Code-AI_skills_plugin-8A2BE2" alt="Claude Code plugin"></a>
  <a href="https://datazoode.github.io/anofox-ml/"><img src="https://img.shields.io/badge/try_it-in_your_browser-2f6f5e" alt="Try it in your browser"></a>
  <img src="https://img.shields.io/badge/status-early%20development-orange.svg" alt="Status: early development">
</p>

A DuckDB extension that brings the [anofox-ml](https://github.com/sipemu/anofox-ml) Rust library into SQL: 196 scikit-learn-compatible estimators, checked against scikit-learn. You fit a model per group with an aggregate, store it as a `BLOB` and predict with it in any later query.

> **Status: early development.** The functions below work and are tested; the API may still change.

> **🤖 Claude Code plugin.** Four [skills](#-claude-code-skills) teach Claude the SQL API, so it writes working `ml_fit_agg`, tuning and evaluation queries from a plain-English request. Install with `/plugin marketplace add DataZooDE/anofox-ml` → `/plugin install anofox-ml@anofox-ml`.


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
- [Claude Code Skills](#-claude-code-skills)
- [API Reference](#-api-reference)
- [Estimators](#-estimators)
- [Behaviour Notes](#-behaviour-notes)
- [Multi-Language Support](#-multi-language-support)
- [Development](#-development)
- [License](#-license)


## Installation

### From the Community Extensions Repository

anofox_ml is a [DuckDB community extension](https://duckdb.org/community_extensions/) for DuckDB v1.5.6 and v2.0.0 on Linux, macOS, Windows and DuckDB-Wasm:

```sql
INSTALL anofox_ml FROM community;
LOAD anofox_ml;
SELECT ml_version();
```

You can also download a prebuilt binary from a GitHub release, or build it from source.

### From a GitHub Release

Every [release](https://github.com/DataZooDE/anofox-ml/releases) has a zip per platform (Linux, macOS and Windows; amd64 / arm64) and DuckDB version, since DuckDB v1.5.6 and v2.0.0 are both supported. The DuckDB version is part of the file name, e.g. `anofox_ml-v0.3.0-duckdb-v1.5.6-linux_amd64.zip` or `…-duckdb-v2.0.0-linux_amd64.zip`. A binary loads only into the DuckDB version it was built for. Unzip it and keep the file name `anofox_ml.duckdb_extension`, because DuckDB derives the entry point from the name. The binaries are not signed, so DuckDB needs `allow_unsigned_extensions`:

```bash
unzip anofox_ml-v0.3.0-duckdb-v1.5.6-linux_amd64.zip
duckdb -unsigned
```

```sql
LOAD './anofox_ml.duckdb_extension';
SELECT ml_version();
```

The setting can only be given when the database is opened, e.g. `duckdb.connect(config={"allow_unsigned_extensions": "true"})` in Python or `dbConnect(duckdb(config = list(allow_unsigned_extensions = "true")))` in R. `SHA256SUMS` in the release lists the checksums. DuckDB-Wasm builds (`wasm_mvp`, `wasm_eh`, `wasm_threads`, for DuckDB 1.5.6 and 2.0.0) ship as `anofox_ml.duckdb_extension.wasm` in the same kind of zip. Changes per version are in [CHANGELOG.md](CHANGELOG.md).

### DuckDB-Wasm

The extension runs in the browser too: **[try it in the playground](https://datazoode.github.io/anofox-ml/)**, DuckDB-Wasm with anofox_ml and example queries, nothing to install. CI runs the SQL test suite on the `wasm_eh` builds in DuckDB-Wasm, under Node for DuckDB 1.5.6 and 2.0 and in headless Chromium through the playground page for 2.0 (`test/wasm`). `wasm_mvp` and `wasm_threads` are built and checked statically: DuckDB-Wasm's threaded runtime cannot load any extension at the moment, DuckDB's own included ([extension-ci-tools#453](https://github.com/duckdb/extension-ci-tools/issues/453)). In the browser everything runs on one thread, so fits that use several threads natively (forests, boosting, tuning) are slower, and memory is limited to the browser tab's heap. `PerpetualRegressor` / `PerpetualClassifier` are not in the wasm builds, because their library always starts its own threads.

### From Source

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-ml.git
cd anofox-ml

GEN=ninja make release       # or: make release -j$(nproc)

# The extension is built to:
# build/release/extension/anofox_ml/anofox_ml.duckdb_extension
# and build/release/duckdb starts a shell with it already loaded.
```

The `duckdb` submodule is pinned to v1.5.6, so a source build targets DuckDB v1.5.6. The same source builds against DuckDB 2.0 (the differences live in `src/include/duckdb_compat.hpp`). For a 2.0 build, check out `duckdb` at `v2.0.0` and `extension-ci-tools` at `v2.0-cyanoptera`; CI builds and tests both.

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

-- predict with the stored models and evaluate per store
SELECT store,
       ml_rmse(units, ml_predict(m.model, [price, promo, dow])) AS rmse,
       ml_r2(units, ml_predict(m.model, [price, promo, dow])) AS r2
FROM test t JOIN models m USING (store)
GROUP BY store ORDER BY store;
-- rmse: store_0 ≈ 1.59, store_1 ≈ 1.70, store_2 ≈ 1.84

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
-- class probabilities, in the order of the stored classes, and their scores
SELECT ml_model_info(model).classes, ml_predict_proba(model, [x1, x2])
FROM churn_model, customers;

SELECT ml_roc_auc(churned, ml_predict_proba(model, [x1, x2])[2]) AS auc,
       ml_log_loss(churned, ml_predict_proba(model, [x1, x2])) AS log_loss,
       ml_f1(churned, ml_predict(model, [x1, x2])) AS f1
FROM churn_model, customers;

-- string class labels
CREATE TABLE iris_model AS SELECT ml_fit_agg('RandomForestClassifier', species, [sepal_length, petal_width]) AS model FROM iris;
SELECT ml_accuracy(species, ml_predict_label(model, [sepal_length, petal_width])) FROM iris, iris_model;

-- unsupervised fit: standardise features, then transform
CREATE TABLE scaler AS SELECT ml_fit_agg('StandardScaler', [price, promo]) AS model FROM train;
SELECT ml_transform(model, [price, promo]) FROM train, scaler;

-- t-SNE / Isomap embeddings of the training rows, one DOUBLE[] per row
SELECT unnest(list(id ORDER BY id)) AS id,
       unnest(ml_fit_transform_agg('TSne', [x1, x2, x3], '{"n_components": 2}' ORDER BY id)) AS xy
FROM points;

-- which features matter (on held-out rows)
SELECT unnest(ml_permutation_importance_agg(model, y, [x1, x2, x3], '{"n_repeats": 10}'), recursive := true)
FROM test, models GROUP BY model;

-- partial dependence of the prediction on x1: average prediction over the data with x1 set to each grid value
SELECT g AS x1, avg(ml_predict(model, [g, x2, x3])) AS partial_dependence
FROM test, models, (SELECT unnest(range(0, 101, 10))::DOUBLE AS g) grid
GROUP BY g ORDER BY g;

-- sample weights and the fitted state
SELECT ml_model_fitted(ml_fit_agg('OlsRegressor', y, [x1, x2], '{}', weight)) FROM train;

-- every estimator and its defaults
SELECT kind, default_params FROM ml_estimators();
```

## 🤖 Claude Code Skills

anofox-ml ships a [Claude Code](https://claude.com/claude-code) plugin with four skills. They cover every function, the option names and the gotchas (row order in per-row aggregates, NaN for imputers, evaluation on held-out rows). Every SQL example in them runs against the built extension (`make check_skills`).

```
/plugin marketplace add DataZooDE/anofox-ml
/plugin install anofox-ml@anofox-ml
```

| Skill | Covers |
|---|---|
| `anofox-ml-modeling` | `ml_fit_agg` per group, the predict functions, params, string labels, weights, inspection, warm start / partial fit, `ml_fit_predict_by` |
| `anofox-ml-features` | scalers and transformers, categorical and text features, imputation, clustering, outlier scores, embeddings, covariance |
| `anofox-ml-evaluation` | metric aggregates, cross-validation splitters, `ml_tune_agg` (grid / random / halving / TPE), feature importance, conformal intervals and sets |
| `anofox-ml-advanced` | voting / stacking / calibration / transformed targets, GLM offsets, multi-target, CCA, Gaussian processes, drift and DLE monitoring |

The skills live in [`plugins/anofox-ml/`](plugins/anofox-ml/). For development, load them with `claude --plugin-dir ./plugins/anofox-ml`.

## 📚 API Reference

| Function | Description |
|---|---|
| `ml_estimators()` | Table of every estimator `kind` with its default parameters (JSON) and `capabilities` (its methods, e.g. `predict_proba`, `transform`, `feature_importances`). `SELECT kind FROM ml_estimators() WHERE list_contains(capabilities, 'predict_proba')` lists the probabilistic classifiers |
| `ml_fit_agg(kind, y, [x1, x2, …] [, params [, weight]])` | Aggregate: fit a model per group, returns a model `BLOB`. `y` is `DOUBLE`, or `VARCHAR` class labels for classifiers. `params` is a JSON object merged into the defaults; `weight` is a per-row sample weight |
| `ml_warm_start_agg(model, y, [x1, x2, …] [, params])` | Aggregate: continue training a stored model on the group's rows: more boosting stages (gradient boosting, HGB, LGBM; raise `n_estimators` in `params`) or more MLP epochs. On the same rows, 100 trees warm-started to 200 equal a 200-tree fit. Targets must be classes / labels the model knows |
| `ml_partial_fit_agg(model, [y,] [x1, x2, …] [, params])` | Aggregate: fold the group's rows into a stored model as one more batch (`partial_fit`). Works for the naive Bayes kinds, SGD, PassiveAggressive and MLP models, and without `y` for StandardScaler, MinMaxScaler, MaxAbsScaler and IncrementalPca. For naive Bayes and the scalers, batches give the same model as one fit. Classes are fixed by the first fit |
| `ml_fit_with_validation_agg(kind, y, [x1, x2, …], is_validation [, params [, weight]])` | `ml_fit_agg` with an explicit validation set: rows with `is_validation = true` are the validation set of early stopping (HistGradientBoosting with `"early_stopping": "Enabled"`), the others train |
| `ml_fit_with_offset_agg(kind, y, [x1, x2, …], offset [, params [, weight]])` | `ml_fit_agg` for the GLMs with an offset per row on the link scale, e.g. `ln(exposure)` for counts: `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor` and `BinomialRegressor` (proportions `y`, trial counts as `weight`). Rows with a NULL offset are skipped |
| `ml_fit_with_group_agg(kind, y, [x1, x2, …], query [, params [, weight]])` | Learning to rank: `query` is a numeric query id per row (`dense_rank()` for strings), and `y` the relevance. For `LgbmRanker` (LambdaRank), and the ranking objectives of `LgbmRegressor` / `CatBoostRegressor`. A query's rows need not be consecutive; `ml_predict` gives the ranking scores |
| `ml_fit_agg(kind, [x1, x2, …] [, params [, weight]])` | The same without a target, for unsupervised estimators (scalers, PCA, clustering, …), and for the kinds fitted on `[X \| Y]` with `y_columns` (`MultiOutputRegressor`, `RegressorChain`, `MultiTask*`, `Cca`, `PlsCanonical`, `PlsSvd`): pass `list_concat(x, y)`; the model then takes `X` alone |
| `ml_predict(model, [x1, x2, …])` | Predict with a model BLOB (`DOUBLE`; the label index for string labels) |
| `ml_predict(model, [x1, x2, …], offset)` | GLM prediction with an offset on the link scale (a model from `ml_fit_with_offset_agg`); `ml_predict(model, x)` predicts at offset 0 |
| `ml_predict_targets(model, [x1, x2, …])` | One prediction per target (`DOUBLE[]`) of the multi-output kinds (`MultiOutputRegressor`, `RegressorChain`, `MultiTaskLasso`, …); `ml_predict` asks for it when a model predicts several targets |
| `ml_predict_std(model, [x1, x2, …])` | Standard deviation of the predictive distribution (`DOUBLE`) of a `GaussianProcessRegressor` |
| `ml_predict_label(model, [x1, x2, …])` | Predicted class as `VARCHAR`: the label of a model fitted on string labels, otherwise the class value as text |
| `ml_predict_proba(model, [x1, x2, …])` | Class probabilities (`DOUBLE[]`), in the order of `ml_model_info(model).classes` |
| `ml_predict_log_proba(model, [x1, x2, …])` | `ln` of the class probabilities (`DOUBLE[]`) |
| `ml_decision_function(model, [x1, x2, …])` | Decision function values (`DOUBLE[]`) |
| `ml_transform(model, [x1, x2, …])` / `ml_inverse_transform(…)` | Transformed features (`DOUBLE[]`); `ml_transform(encoder, [VARCHAR, …])` for the categorical encoders |
| `ml_transform_y(model, [y1, y2, …])` | Scores of the second block (`DOUBLE[]`) of `Cca`, `PlsCanonical` and `PlsSvd`, whose `ml_transform` gives the X scores |
| `ml_score_samples(model, [x1, x2, …])` | Per-sample scores, e.g. outlier scores or log-likelihoods (`DOUBLE[]`) |
| `ml_staged_predict(model, [x1, x2, …])` | Boosting kinds (gradient boosting, HGB, LGBM): the prediction after each stage (`DOUBLE[]`; the class or label index per stage for classifiers). The last entry equals `ml_predict` |
| `ml_cluster_agg(kind, [x1, x2, …] [, params [, weight]])` | Aggregate: fit a clustering (or outlier) estimator on the group and return one label per row (`BIGINT[]`; `-1` is noise for DBSCAN-like methods) |
| `ml_outlier_score_agg(kind, [x1, x2, …] [, params [, weight]])` | Aggregate: fit an outlier detector on the group and return one score per row (`DOUBLE[]`; lower is more anomalous, as in scikit-learn) |
| `ml_fit_transform_agg(kind, [y,] [x1, x2, …] [, params [, weight]])` | Aggregate: fit on the group and return each row's output (`DOUBLE[][]`): the training embedding of t-SNE, Isomap, LLE and MDS, TargetEncoder's cross-fitted encoding, or `transform` of the training rows |
| `ml_model_info(model)` | `STRUCT` of kind, `n_features`, `classes`, `capabilities` (supported methods), `params` (JSON), `format_version`, `crate_version`, `labels` (string class labels), and the training statistics `n_iter`, `inertia` (KMeans) and `oob_score` (forests with `oob_score: true`), NULL where the model has none |
| `ml_permutation_importance_agg(model, y, [x1, x2, …] [, options])` | Aggregate over evaluation rows: `STRUCT(feature, importance_mean, importance_std)[]`, the drop in a metric when feature `feature` (1-based, as in `x[feature]`) is shuffled. Options: `metric` (any metric aggregate name; default `accuracy` for classifiers, `r2` otherwise; losses are negated so higher = more important), `metric_options`, `n_repeats` (5), `seed` |
| `ml_kneighbors(model, [x1, x2, …] [, k])` | `STRUCT(indices BIGINT[], distances DOUBLE[])` of the `k` nearest training rows of a `KnnClassifier` / `KnnRegressor` (default: its `n_neighbors`), nearest first. `indices` are 0-based positions in the fit's input order, so fit with `ORDER BY` to map them back to rows |
| `ml_rfe_agg(kind, y, [x1, x2, …] [, params [, options]])` | Aggregate: recursive feature elimination, as scikit-learn's `RFE`. Each round fits `kind` and drops the `step` weakest features (by `feature_importances`, or by the absolute coefficients) until `n_features_to_select` remain (default: half; counts or fractions). Returns `STRUCT(support BOOLEAN[], ranking INTEGER[], model BLOB)`; `model` is fitted on the selected features, in their original order |
| `ml_mutual_info_agg(y, [x1, x2, …] [, options])` | Aggregate: mutual information (nats) between each feature and `y`, as scikit-learn's `mutual_info_regression` / `mutual_info_classif` (`DOUBLE[]`). `y` may be `VARCHAR`. Options: `discrete_target`, `discrete_features` (`"auto"`, `true` or 1-based positions), `n_neighbors` (3), `seed` |
| `ml_covariance_agg([x1, x2, …] [, options])` | Aggregate: `STRUCT(covariance DOUBLE[][], shrinkage, location DOUBLE[])`, as scikit-learn. `method` = `"ledoit_wolf"` (default), `"oas"`, `"empirical"`, `"shrunk"` (with `shrinkage`, default 0.1) or `"mcd"` (robust minimum covariance determinant; `support_fraction`, `seed`; `shrinkage` is NULL) |
| `ml_covariance(model)` | `STRUCT(location, covariance, precision)` of a fitted covariance estimator (`EmpiricalCovariance`, `ShrunkCovariance`, `LedoitWolf`, `Oas`, `MinCovDet`, `EllipticEnvelope`) |
| `ml_mahalanobis(model, [x1, x2, …])` | Squared Mahalanobis distance of a row to a fitted covariance estimator |
| `ml_estimate_bandwidth_agg([x1, x2, …] [, options])` | Aggregate: a MeanShift bandwidth (scikit-learn `estimate_bandwidth`); options `quantile` (0.3), `n_samples`, `seed` |
| `ml_multiple_impute_agg(id, [x1, x2, …] [, options] ORDER BY id)` | Aggregate: multiple imputation as Hmisc's `aregImpute` (additive regression + predictive mean matching). A NULL element is a missing value; returns `STRUCT(imputation, id, x DOUBLE[])[]`, `n_impute` completed copies of the rows. Options: `n_impute` (5), `nk` (3), `tlinear`, `burnin`, `match` (`"weighted"` / `"closest"`), `fweighted`, `kinds` (per column `"numeric"` / `"linear"` / `"categorical"`), `seed` (1, as R's `set.seed(1)`) |
| `ml_spectral_embedding_agg(src, dst [, weight [, options]])` | Aggregate over the edges of an undirected graph: `STRUCT(node, embedding DOUBLE[])[]`, scikit-learn's `spectral_embedding` (options `n_components` (2), `drop_first` (true)) |
| `ml_model_fitted(model)` | The fitted state as JSON: coefficients, centroids, components, trees, … (arrays as `{"v", "dim", "data"}`, e.g. `json_extract(ml_model_fitted(m), '$.coef.data')`) |
| `ml_coefficients(model)` | `STRUCT(coef DOUBLE[][], intercept DOUBLE[])` of a linear model (one `coef` row per class for multiclass models) |
| `ml_cluster_centers(model)` | `DOUBLE[][]`: centroids / cluster centres / mixture means |
| `ml_explained_variance_ratio(model)` | `DOUBLE[]` of PCA-like decompositions |
| `ml_feature_importances(model)` | `DOUBLE[]`, one impurity-based importance per feature, summing to 1: decision trees, random forests, extra trees, gradient boosting, histogram gradient boosting, LGBM and bagging. For any model, see `ml_permutation_importance_agg` |
| `ml_model_kind(model)` | The estimator kind stored in a model BLOB (without decoding the model) |
| `ml_version()` | Version of the bundled anofox-ml FFI layer |

### Categorical Features

`OneHotEncoder` and `OrdinalEncoder` work on `VARCHAR` columns, as in scikit-learn: `ml_fit_agg(kind, [col_a, col_b, …] [, params])` learns the sorted categories of every column, and `ml_transform(encoder, [col_a, col_b, …])` returns a `DOUBLE[]` to combine with numeric features via `list_concat`.

| Params | Description |
|---|---|
| `handle_unknown` | `"error"` (default) or `"ignore"`: an unseen category encodes as all zeros (one-hot) or `unknown_value` (ordinal, default -1) |
| `drop` | One-hot only: `"first"` drops each column's first category, `"if_binary"` only for two-category columns |

Rows with a NULL value are skipped when fitting and encode as NULL; use `coalesce(col, 'missing')` to keep them as a category.

```sql
CREATE TABLE enc AS SELECT ml_fit_agg('OneHotEncoder', [city, channel], '{"handle_unknown": "ignore"}') AS e FROM train;
CREATE TABLE model AS
SELECT ml_fit_agg('HistGradientBoostingRegressor', sales, list_concat(ml_transform(e, [city, channel]), [price, promo])) AS m
FROM train, enc;
SELECT ml_predict(m, list_concat(ml_transform(e, [city, channel]), [price, promo])) FROM test, enc, model;
```

### Text Features

| Function | Description |
|---|---|
| `ml_hashing_vectorize(text, n_features [, options])` | scikit-learn's `HashingVectorizer`: unigram tokens (2+ word characters) hashed into `n_features` buckets, as a dense `DOUBLE[]` to feed estimators directly. Options: `alternate_sign` (true), `norm` (`"l2"` or `null` for counts) |
| `ml_hashing_vectorize_sparse(text, n_features [, options])` | The same as `STRUCT(indices INTEGER[], values DOUBLE[])`, buckets 0-based, for large `n_features` |
| `ml_fit_agg(kind, text [, params])` | Aggregate: fit a text vectorizer on the group's documents (one `VARCHAR` per row; NULL rows are skipped): `CountVectorizer`, `TfidfVectorizer` (options `min_df`, `max_df_frac`, `norm_l2`) or `HashingVectorizer` (`n_features`, `alternate_sign`, `norm_l2`). The vocabulary (sorted, as scikit-learn's `get_feature_names_out`) and `idf` are in `ml_model_fitted` |
| `ml_transform(model, text)` | Encode a document with a fitted text vectorizer: a dense `DOUBLE[]`, one entry per vocabulary word (words outside the vocabulary are ignored). Matches scikit-learn 1.8 on the default analyzer (lowercase, tokens of 2+ word characters) |

```sql
CREATE TABLE spam_model AS
SELECT ml_fit_agg('MultinomialNB', label, ml_hashing_vectorize(body, 1024, '{"alternate_sign": false}')) AS model FROM mails;
SELECT ml_predict_label(model, ml_hashing_vectorize(body, 1024, '{"alternate_sign": false}')) FROM new_mails, spam_model;
```

For TF-IDF over a vocabulary, fit a `TfidfVectorizer` on the documents. The output is dense, one column per word, so keep the vocabulary small with `min_df` / `max_df_frac` on large corpora:

```sql
CREATE TABLE tfidf AS SELECT ml_fit_agg('TfidfVectorizer', body, '{"min_df": 5}') AS vec FROM mails;
CREATE TABLE clf AS SELECT vec, ml_fit_agg('LinearSvc', label, ml_transform(vec, body)) AS model FROM mails, tfidf GROUP BY vec;
SELECT ml_predict_label(model, ml_transform(vec, body)) FROM new_mails, clf;
```

Without a vocabulary (any corpus size), hash raw counts (`"norm": null`) and fit a `TfidfTransformer` on them. This gives scikit-learn's `HashingVectorizer` + `TfidfTransformer` exactly:

```sql
CREATE MACRO counts(t) AS ml_hashing_vectorize(t, 4096, '{"alternate_sign": false, "norm": null}');
CREATE TABLE tfidf AS SELECT ml_fit_agg('TfidfTransformer', counts(body)) AS idf FROM mails;
CREATE TABLE clf AS SELECT idf, ml_fit_agg('LinearSvc', label, ml_transform(idf, counts(body))) AS model FROM mails, tfidf GROUP BY idf;
SELECT ml_predict_label(model, ml_transform(idf, counts(body))) FROM new_mails, clf;
```

### Hyperparameter Tuning

`ml_tune_agg(kind, y, [x1, x2, …], search_space [, options])` searches hyperparameters by k-fold cross-validation and returns `STRUCT(best_params, best_score, model, trials)`. `model` is the best candidate refitted on all rows, ready for `ml_predict`, and `trials` is a JSON array of every candidate's mean / std score (or its error). Per `GROUP BY` group, one search each.

- **Search space** (JSON object, one entry per parameter): a list of values (grid / categorical), `{"int": [lo, hi]}`, `{"float": [lo, hi]}` (both with optional `"log": true`) or `{"categorical": […]}`.
- **Options:** `search` (`"random"` default, `"grid"`, successive halving `"halving"` / `"halving_grid"`: all candidates on a small seeded subsample, the best `1/factor` (default 3) kept and the rows grown by `factor` each round, or `"tpe"`: a tree-structured Parzen estimator as in Optuna, `n_startup_trials` (10) random trials and then each trial proposed from the earlier scores, `n_ei_candidates` (24) draws per proposal), `n_trials` (20), `cv` (5 folds; stratified for classifiers unless `"stratify": false`), `metric` (any metric aggregate name; default `accuracy` / `r2`; losses are minimised), `metric_options`, `params` (fixed for every candidate), `seed`.

Candidates are drawn from `seed` up front (TPE: one after the other, from `seed` and the earlier scores) and every candidate × fold is scored in parallel, so the result is the same whatever the thread count. TPE trials run one at a time, so only the folds of a trial run in parallel.

```sql
SELECT store, t.best_params, t.best_score
FROM (SELECT store, ml_tune_agg('HistGradientBoostingRegressor', sales, [price, promo, dow],
        '{"max_depth": {"int": [2, 12]}, "learning_rate": {"float": [0.01, 0.3], "log": true}}',
        '{"n_trials": 30, "metric": "rmse"}') AS t
      FROM train GROUP BY store);
```

### Conformal Prediction

Two ways to get prediction intervals (regressors) and prediction sets (classifiers). Coverage is guaranteed to be at least `1 − alpha` when the calibration and new rows are exchangeable.

- **Calibrate any fitted model** on held-out rows with `ml_conformalize_agg` (split conformal), then ask for intervals or sets at any `alpha`.
- **Fit a conformal kind**, which holds out its own calibration rows or cross-fits when it is fitted: `SplitConformalRegressor`, `CrossConformalRegressor` (CV+ / jackknife+), `NormalizedConformalRegressor` (widths follow the local error scale), `ConformalizedQuantileRegressor` (CQR; `alpha` is fixed at fit time), `MondrianConformalRegressor` (coverage per group), `SplitConformalClassifier` (LAC / APS / RAPS sets) and `MondrianConformalClassifier`. They take the base model as `estimator` + `estimator_params`. The Mondrian kinds read the group id, a non-negative integer, from the feature at `group_column`.

| Function | Description |
|---|---|
| `ml_conformalize_agg(model, y, [x1, x2, …])` | Aggregate over calibration rows: a copy of `model` with its conformity scores (absolute residuals for regressors, `1 − p(true class)` for classifiers). `y` may be `VARCHAR` for a model fitted on string labels |
| `ml_predict_interval(model, [x1, x2, …], alpha)` | `STRUCT(lower, upper)`: for a calibrated model, prediction ± the conformal quantile of the residuals (infinite when there are too few calibration rows for `alpha`); for a conformal kind, its own interval; for `OlsRegressor` / `RidgeRegressor` fitted with `{"inference": true}`, the classical prediction interval at level `1 − alpha` |
| `ml_predict_set(model, [x1, x2, …], alpha)` | `VARCHAR[]` of the classes whose score is within the conformal quantile (calibrated model) or in the conformal kind's set |
| `ml_conformal_margin(y, y_pred, alpha)` | Aggregate: the conformal quantile of `abs(y − y_pred)`, model-free |
| `ml_set_coverage(y, prediction_set)` / `ml_mean_set_size(prediction_set)` | Aggregates: the share of rows whose set contains `y`, and the mean set size |
| `ml_mean_pinball_loss(y, q, alpha)` | Aggregate: pinball (quantile) loss of predictions `q` of the `alpha` quantile |

```sql
CREATE TABLE model AS SELECT ml_fit_agg('HistGradientBoostingRegressor', y, [x1, x2]) AS m FROM train;
CREATE TABLE calibrated AS SELECT ml_conformalize_agg(m, y, [x1, x2]) AS m FROM calibration, model GROUP BY ALL;

SELECT ml_predict(m, [x1, x2]) AS forecast,
       ml_predict_interval(m, [x1, x2], 0.1) AS pi90
FROM test, calibrated;

-- check it: coverage and interval score
SELECT ml_coverage(y, pi90.lower, pi90.upper), ml_interval_score(y, pi90.lower, pi90.upper, 0.1)
FROM (SELECT y, ml_predict_interval(m, [x1, x2], 0.1) AS pi90 FROM test, calibrated);

-- or in one step: CV+ intervals around gradient boosting, one model per store
CREATE TABLE cv_models AS
SELECT store, ml_fit_agg('CrossConformalRegressor', y, [x1, x2],
       '{"estimator": "HistGradientBoostingRegressor", "estimator_params": {"n_estimators": 200}, "cv": 5}') AS m
FROM train GROUP BY store;
```

### Table Macros

One-shot workflows over a table (name as a string) and its columns. They keep every source column and add the result columns. Pass a constant such as `1` as `group_col` for a single model over the whole table. `id_col` must be unique within a group; it aligns per-row results and fixes the row order.

| Macro | Adds |
|---|---|
| `ml_fit_predict_by(source, group_col, y_col, x_cols, kind [, params := …])` | `prediction`: one model per group, fitted on the rows with a target, predicting every row (also those without one) |
| `ml_fit_predict_proba_by(source, group_col, y_col, x_cols, kind [, params := …])` | `prediction`, `label`, `proba DOUBLE[]`, `classes` and `labels` |
| `ml_cross_val_predict_by(source, group_col, id_col, y_col, x_cols, kind [, params := …, method := 'kfold', options := '{}'])` | `split` and the out-of-fold `prediction` of every labelled row (methods as in `ml_cv_splits_agg`) |
| `ml_domain_classifier_drift_by(source, chunk_col, id_col, x_cols, reference [, kind := 'HistGradientBoostingClassifier', params := '{}', folds := 5, threshold := 0.55, seed := 0])` | Multivariate drift per chunk, as evidently's "model" method and NannyML's domain classifier. A classifier learns to tell the chunk's rows from the `reference` chunk's, and the stratified out-of-fold ROC AUC is about 0.5 without drift. Returns `chunk, n_reference, n_rows, auc, threshold, drifted` |
| `ml_dle_estimate_by(reference, analysis, chunk_col, y_col, pred_col, x_cols [, kind := 'LgbmRegressor', params := '{}'])` | Direct loss estimation, as NannyML's DLE. It estimates a regression model's MAE and RMSE per chunk of `analysis` without labels: two models learn the absolute and the squared error on the labelled `reference` rows, from `x_cols` plus the prediction. Returns `chunk, n_rows, estimated_mae, mae_sampling_error, estimated_rmse, rmse_sampling_error` |
| `ml_cluster_by(source, group_col, id_col, x_cols, kind [, params := …])` | `cluster` |
| `ml_outlier_score_by(source, group_col, id_col, x_cols, kind [, params := …])` | `outlier_score` |
| `ml_fit_transform_by(source, group_col, id_col, x_cols, kind [, params := …])` | `features DOUBLE[]` (embeddings, encodings, transforms) |

```sql
-- train and forecast in one call: rows with NULL sales get a forecast
FROM ml_fit_predict_by('sales', store, sales, [price, promo, dow], 'HistGradientBoostingRegressor',
                       params := '{"n_estimators": 200}');

-- 5-fold cross-validated RMSE per store
SELECT store, ml_rmse(sales, prediction) AS cv_rmse
FROM ml_cross_val_predict_by('sales', store, id, sales, [price, promo, dow], 'RandomForestRegressor',
                             options := '{"n_splits": 5, "shuffle": true}')
GROUP BY store;

-- clusters and outlier scores per row
FROM ml_cluster_by('points', region, id, [x, y], 'Hdbscan', params := '{"min_cluster_size": 10}');
FROM ml_outlier_score_by('readings', machine, id, [temp, vibration], 'IsolationForest');
```

### Metrics

Aggregates over `(y_true, y_pred)`: no model is needed, so they score any prediction, per group or over the whole table. Rows with a NULL argument are skipped and an empty group gives NULL. They are computed by anofox-ml's metrics crate and match scikit-learn.

| Function | Description |
|---|---|
| `ml_r2`, `ml_mse`, `ml_rmse`, `ml_mae`, `ml_mape`, `ml_median_absolute_error`, `ml_max_error`, `ml_explained_variance`, `ml_mean_squared_log_error` `(y, y_pred)` | Regression |
| `ml_accuracy`, `ml_precision`, `ml_recall`, `ml_f1`, `ml_fbeta` `(y, y_pred [, options [, weight]])` | Classification on labels (`DOUBLE` or `VARCHAR`). `options` JSON: `average` (`binary` default, `macro`, `micro`, `weighted`), `pos_label`, `zero_division` (`0`, `1`, `"nan"`), `labels`, `beta` |
| `ml_balanced_accuracy`, `ml_matthews_corrcoef` (binary), `ml_cohen_kappa` `(y, y_pred)` | Classification on labels (`DOUBLE` or `VARCHAR`) |
| `ml_roc_auc`, `ml_log_loss` `(y, score DOUBLE \| proba DOUBLE[] [, options [, weight]])` | Binary score or per-class probabilities (columns in sorted label order, or `labels`). ROC AUC options: `multi_class` (`ovr`, `ovo`), `average` |
| `ml_brier_score(y, p [, options [, weight]])`, `ml_average_precision(y, score)` | Binary probabilities and scores |
| `ml_adjusted_rand`, `ml_normalized_mutual_info` `(labels_true, labels_pred)`, `ml_silhouette(x DOUBLE[], label)` | Clustering |
| `ml_coverage(y, lower, upper)`, `ml_interval_score(y, lower, upper, alpha)` | Prediction intervals |

### Cross-Validation

`ml_cv_splits_agg(id, method, options [, y [, groups]] ORDER BY …)` returns one `STRUCT(id, split, role)` per row and split, where `role` is `'train'` or `'test'`. `unnest(…, recursive := true)` turns that into a fold table to join back to the data. The `ORDER BY` inside the aggregate fixes the row order, which is the time order for the time-series methods.

| Method | Needs | Options (scikit-learn names and defaults) |
|---|---|---|
| `kfold`, `repeated_kfold` | — | `n_splits` (5), `shuffle`, `seed`, `n_repeats` (10) |
| `shuffle_split` | — | `n_splits` (10), `test_size` (0.1), `seed` |
| `leave_one_out`, `leave_p_out` | — | `p` |
| `time_series` | — | `n_splits` (5), `test_size` (rows), `gap`, `max_train_size`, `min_train_size` |
| `stratified_kfold`, `repeated_stratified_kfold`, `stratified_shuffle_split` | `y` | `n_splits`, `n_repeats`, `test_size`, `seed` (the stratified folds are always shuffled with `seed`) |
| `stratified_kfold_continuous` | `y` | `n_splits`, `n_bins` (10), `seed` |
| `group_kfold`, `leave_one_group_out`, `leave_p_groups_out`, `group_shuffle_split` | `groups` | `n_splits`, `shuffle`, `seed`, `p`, `test_size` (0.2), `train_size` |
| `group_time_series` | `groups` (time keys, in `ORDER BY` order) | as `time_series` |
| `stratified_group_kfold` | `y`, `groups` | `n_splits`, `shuffle`, `seed` |

Pass `NULL` for `y` when a method only needs `groups`. Rows with a NULL id, or without the `y` / group the method needs, are left out.

For labels that span time (a return over the next days, a churn window), `ml_purged_cv_splits_agg(id, t_start, t_end, method, options ORDER BY t_start)` purges training rows whose label interval `[t_start, t_end]` overlaps the test span, and drops `embargo` rows after it (López de Prado). Times are `DOUBLE`, e.g. `epoch(ts)`.

| Method | Options |
|---|---|
| `purged_kfold` | `n_splits` (5), `embargo` (0 rows) |
| `combinatorial_purged` | `n_groups` (6), `n_test_groups` (2), `embargo`. Test entries carry the backtest `path` they belong to (one entry per path); `WHERE role = 'test' AND path = k` is a complete out-of-sample backtest |

For spatial data, `ml_spatial_cv_splits_agg(id, coords DOUBLE[], method, options ORDER BY id)` keeps nearby points out of each other's training data:

| Method | Options |
|---|---|
| `spatial_block_kfold` | `block_size` (cell width in coordinate units, required), `n_splits` (5), `seed`; whole cells share a fold |
| `kfold` | `n_splits`, `seed` (usually with a `buffer`) |
| `buffered_loo` | `buffer` (radius, required): one split per point, neighbours within the radius left out of training |

`buffer` (any method) removes training points within that distance of a test point of the same split, and `metric` is `"euclidean"` (default) or `"haversine"` (km, coordinates `[lat, lon]` in degrees).

```sql
-- 5-fold cross-validated RMSE of a random forest
WITH folds AS (
    SELECT unnest(ml_cv_splits_agg(id, 'kfold', '{"n_splits": 5, "shuffle": true}' ORDER BY id), recursive := true)
    FROM train),
f AS (SELECT * FROM folds JOIN train USING (id)),
models AS (
    SELECT split, ml_fit_agg('RandomForestRegressor', y, [x1, x2]) AS model
    FROM f WHERE role = 'train' GROUP BY split)
SELECT avg(rmse), stddev(rmse) FROM (
    SELECT split, ml_rmse(y, ml_predict(model, [x1, x2])) AS rmse
    FROM f JOIN models USING (split) WHERE role = 'test' GROUP BY split);
```

### Estimator Recipes

**Wrapping another estimator.** Meta-estimators and conformal kinds take the base model as a kind name plus its parameters, nested in the JSON:

```sql
-- one-vs-rest SVMs over string labels
SELECT ml_fit_agg('OneVsRestClassifier', churn, [a, b],
       '{"estimator": "Svc", "estimator_params": {"c": 10.0}}') AS model FROM customers;

-- a decision threshold tuned by cross-validation for F1 (binary targets), or a fixed one
SELECT ml_fit_agg('TunedThresholdClassifierCV', churn, [a, b], '{"scoring": "F1"}') FROM customers;
SELECT ml_fit_agg('FixedThresholdClassifier', churn, [a, b],
       '{"estimator": "HistGradientBoostingClassifier", "threshold": 0.3}') FROM customers;
```

The defaults (`SELECT default_params FROM ml_estimators() WHERE kind = '…'`) show every option. `TunedThresholdClassifierCV` scores with `BalancedAccuracy` (default), `Accuracy`, `F1`, `Precision` or `Recall`.

**Ensembles of different models.** `VotingRegressor` / `VotingClassifier` and `StackingRegressor` / `StackingClassifier` take a list of `{estimator, estimator_params}` specs, as scikit-learn's `estimators=[...]`. Voting averages the members' predictions (with optional `weights`; the classifier by majority, `"voting": "hard"`, or by mean probability, `"soft"`). Stacking fits `final_estimator` on the members' `cv`-fold out-of-fold predictions (KFold / StratifiedKFold without shuffling; `passthrough` adds the features, `stack_method` picks `predict_proba` / `decision_function` / `predict`):

```sql
SELECT ml_fit_agg('VotingClassifier', churn, [a, b, c], '{"voting": "soft", "weights": [1, 2],
       "estimators": [{"estimator": "GaussianNB", "estimator_params": {}},
                      {"estimator": "HistGradientBoostingClassifier", "estimator_params": {"n_estimators": 200}}]}') AS model
FROM customers;

SELECT ml_fit_agg('StackingRegressor', units, [price, promo], '{"cv": 5,
       "estimators": [{"estimator": "RidgeRegressor", "estimator_params": {}}, {"estimator": "RandomForestRegressor", "estimator_params": {}}],
       "final_estimator": {"estimator": "OlsRegressor", "estimator_params": {}}}') AS model
FROM sales;
```

**Calibrated probabilities.** `CalibratedClassifierCV` calibrates a classifier's decision function (else its probabilities) on held-out folds: `"method": "sigmoid"` (Platt, default) or `"isotonic"`, one-vs-rest for more than two classes, one calibrated model per fold averaged (`"ensemble": true`) or one model on all rows. It turns margin classifiers such as `LinearSvc` (the default `estimator`) into probabilistic ones:

```sql
SELECT ml_predict_proba(model, [a, b]) FROM test,
       (SELECT ml_fit_agg('CalibratedClassifierCV', label, [a, b], '{"estimator": "GaussianNB", "method": "isotonic"}') AS model FROM train);
```

**Transformed targets.** `TransformedTargetRegressor` fits `regressor` on `func(y)` (`"log"`, `"log1p"`, `"sqrt"`, `"identity"`) or on `y` transformed by a `transformer` kind (`StandardScaler`, `PowerTransformer`, `QuantileTransformer`, …) and maps its predictions back, e.g. a log-linear model for skewed, positive targets:

```sql
SELECT ml_fit_agg('TransformedTargetRegressor', revenue, [x1, x2],
       '{"func": "log", "regressor": {"estimator": "RidgeRegressor", "estimator_params": {"lambda": 0.1}}}') AS model
FROM shops;
```

**Categorical features in CatBoost.** `CatBoostRegressor` / `CatBoostClassifier` build target statistics for the columns listed in `cat_features` (0-based indices into the feature list). Encode string categories as integer codes first (`dense_rank() OVER (ORDER BY store) - 1`, or `ml_fit_agg('OrdinalEncoder', …)`). They use CatBoost's own defaults: random permutations, MVS bootstrap and score noise. For a fit that depends only on the row order, pass `{"has_time": true, "random_strength": 0, "bootstrap_type": "No"}` and order the rows inside the aggregate:

```sql
SELECT ml_fit_agg('CatBoostRegressor', units, [price, store_code], '{"cat_features": [1]}' ORDER BY day) FROM sales;
```

**Robust outlier screening per group.** `MinCovDet` estimates location and covariance from the bulk of the data, so a few extreme rows don't inflate it; flag rows by their squared Mahalanobis distance (about χ²(p) for Gaussian data):

```sql
WITH m AS (SELECT sensor, ml_fit_agg('MinCovDet', [temp, pressure]) AS model FROM readings GROUP BY sensor)
SELECT r.*, ml_mahalanobis(model, [temp, pressure]) > 13.8 AS outlier  -- χ²(2) at 0.999
FROM readings r JOIN m USING (sensor);
```

**Densities and robust outliers.** `KernelDensity` and `FactorAnalysis` give per-row log-likelihoods through `ml_score_samples`. `EllipticEnvelope` (robust covariance) scores rows with `ml_outlier_score_agg` like the other detectors (lower = more anomalous):

```sql
SELECT ml_score_samples(ml_fit_agg('KernelDensity', [a, b], '{"bandwidth": {"Fixed": 1.0}}'), [1.0, 3.0]) FROM train;
```

**Learning curves and early stopping.** `ml_staged_predict` scores every boosting stage at once. `ml_fit_with_validation_agg` stops on a validation set you choose, for example the last weeks of a series instead of a random split:

```sql
-- validation RMSE per stage
CREATE TABLE model AS SELECT ml_fit_agg('HistGradientBoostingRegressor', y, [x1, x2], '{"n_estimators": 200}') AS m FROM train;
SELECT stage, ml_rmse(y, p[stage]) AS rmse
FROM (SELECT y, ml_staged_predict(m, [x1, x2]) AS p FROM test, model), range(1, 201) r(stage)
GROUP BY stage ORDER BY stage;

-- per store: train on the history, stop on the last 28 days
SELECT store, ml_fit_with_validation_agg('HistGradientBoostingRegressor', units, [price, dow],
       day > max_day - 28, '{"n_estimators": 1000, "early_stopping": "Enabled"}') AS model
FROM (SELECT *, max(day) OVER (PARTITION BY store) AS max_day FROM sales) GROUP BY store;
```

**Adding trees to a stored model.** Instead of refitting from scratch, continue boosting from the stored BLOB with this week's rows:

```sql
UPDATE models SET model = w.model
FROM (SELECT store, ml_warm_start_agg(m.model, units, [price, promo, dow], '{"n_estimators": 300}') AS model
      FROM new_rows JOIN models m USING (store) GROUP BY store, m.model) w
WHERE models.store = w.store;
```

**Selecting features.** `ml_rfe_agg` eliminates features recursively. To select by a threshold on importances (scikit-learn's `SelectFromModel`), filter `ml_feature_importances` in SQL:

```sql
SELECT ml_rfe_agg('OlsRegressor', y, [x1, x2, x3, x4, x5], '{}', '{"n_features_to_select": 2}').support FROM train;

-- SelectFromModel(threshold="mean")
SELECT list_filter(range(1, len(fi) + 1), lambda j: fi[j] >= list_avg(fi)) AS selected
FROM (SELECT ml_feature_importances(ml_fit_agg('RandomForestRegressor', y, [x1, x2, x3, x4, x5])) AS fi FROM train);
```

**Feature selection kinds.** `Rfe`, `Rfecv` and `SequentialFeatureSelector` are estimator kinds with a nested `estimator` / `estimator_params`, matching scikit-learn. `ml_transform` keeps the selected columns, the support mask (and `ranking`; `Rfecv` also the CV scores) is in `ml_model_fitted`, and `Rfe` / `Rfecv` predict with the estimator refitted on the selection. `Rfecv` picks the number of features by CV; `SequentialFeatureSelector` adds (`"direction": "forward"`) or drops features greedily by CV score:

```sql
WITH s AS (SELECT ml_fit_agg('Rfecv', y, [x1, x2, x3, x4], '{"estimator": "RandomForestRegressor", "cv": 5}') AS sel FROM train)
SELECT ml_transform(sel, [x1, x2, x3, x4]) AS selected, ml_predict(sel, [x1, x2, x3, x4]) FROM test, s;
```

**Exposure and offsets.** Counts observed over different exposures (time at risk, policy years) are modelled as a rate with `ln(exposure)` as the offset; predict the expected count for a given exposure, or the rate with `ml_predict(model, x)`:

```sql
WITH m AS (SELECT ml_fit_with_offset_agg('PoissonRegressor', claims, [age, region_code], ln(policy_years), '{"alpha": 0}') AS model FROM policies)
SELECT ml_predict(model, [age, region_code], ln(policy_years)) AS expected_claims,
       ml_predict(model, [age, region_code]) AS claims_per_year
FROM new_policies, m;
```

**Canonical correlation.** `Cca` (and `PlsCanonical`, `PlsSvd`) fit on both blocks at once: concatenate X and Y and give the width of Y as `y_columns`. `ml_transform` scores the X block, `ml_transform_y` the Y block:

```sql
WITH m AS (SELECT ml_fit_agg('Cca', [x1, x2, x3, y1, y2], '{"n_components": 2, "y_columns": 2}') AS cca FROM t)
SELECT ml_transform(cca, [x1, x2, x3]) AS x_scores, ml_transform_y(cca, [y1, y2]) AS y_scores FROM t, m;
```

**Several targets at once.** The multi-output kinds fit on `[X | Y]`: append the targets to the features and give their count as `y_columns` (default 2). `MultiOutputRegressor` fits one nested `estimator` per target, `RegressorChain` feeds earlier predictions to later targets (`order`), `MultiTaskLasso` / `MultiTaskElasticNet` (and their CV kinds) share the sparsity pattern across targets. `ml_predict_targets` returns one value per target:

```sql
WITH m AS (SELECT ml_fit_agg('MultiOutputRegressor', [price, promo, units, revenue],
                  '{"estimator": "HistGradientBoostingRegressor", "y_columns": 2}') AS model FROM sales)
SELECT ml_predict_targets(model, [price, promo]) AS units_and_revenue FROM plan, m;
```

**Gaussian process uncertainty.** `GaussianProcessRegressor` (scikit-learn's default `1.0 * RBF(1.0)`, fixed hyperparameters) gives a predictive standard deviation next to its mean: small at the training points, approaching the prior's (1.0) far from them:

```sql
SELECT x, ml_predict(gp, [x]) AS mean, ml_predict(gp, [x]) + 1.96 * ml_predict_std(gp, [x]) AS upper95
FROM grid, (SELECT ml_fit_agg('GaussianProcessRegressor', y, [x]) AS gp FROM train);
```

**Classical linear fits.** `OlsRegressor` and `RidgeRegressor` with `{"inference": true}` keep their full fit: `ml_predict_interval(model, x, alpha)` gives prediction intervals at level `1 − alpha` (no calibration set needed, but they assume Gaussian, homoscedastic errors), and `ml_model_fitted(model)` holds the coefficient table under `summary` (`coefficients`, `std_errors`, `statistics`, `p_values`, `conf_lower` / `conf_upper` at `confidence_level`, and the intercept's):

```sql
WITH m AS (SELECT ml_fit_agg('OlsRegressor', y, [x1, x2], '{"inference": true}') AS model FROM train)
SELECT ml_predict_interval(model, [x1, x2], 0.05) AS pi95,
       json_extract(ml_model_fitted(model), '$.summary.p_values.data') AS p_values
FROM test, m;
```

**Multivariate drift.** `ml_domain_classifier_drift_by` flags chunks whose rows a classifier can tell apart from a reference chunk. To see which features drift, fit the same classifier on reference + chunk and read its importances:

```sql
FROM ml_domain_classifier_drift_by('events', week, id, [latency, size, retries], '2026-W40');

SELECT ml_feature_importances(ml_fit_agg('HistGradientBoostingClassifier', (week = '2026-W45')::INT::DOUBLE, [latency, size, retries]))
FROM events WHERE week IN ('2026-W40', '2026-W45');
```

**Performance without labels.** `ml_dle_estimate_by` estimates MAE / RMSE of a deployed regression model per chunk while the labels are not in yet. To fit the error models once and reuse them, run the same steps by hand:

```sql
FROM ml_dle_estimate_by('reference', 'production', week, y, forecast, [price, promo, dow]);

-- the same by hand: store the error model, estimate later
CREATE TABLE dle AS SELECT ml_fit_agg('LgbmRegressor', abs(y - forecast), [price, promo, dow, forecast]) AS mae_model FROM reference;
SELECT week, avg(greatest(ml_predict(mae_model, [price, promo, dow, forecast]), 0)) AS estimated_mae FROM production, dle GROUP BY week;
```

**Learning to rank.** `ml_fit_with_group_agg` fits a ranker on relevance labels per query (search results, recommendations); `ml_predict` scores each row, and ordering by the score within a query is the ranking:

```sql
CREATE TABLE ranker AS
SELECT ml_fit_with_group_agg('LgbmRanker', relevance, [x1, x2], query_id, '{"n_estimators": 50}') AS m FROM search_log;
SELECT query_id, row_number() OVER (PARTITION BY query_id ORDER BY ml_predict(m, [x1, x2]) DESC) AS position, x1, x2
FROM search_log, ranker ORDER BY query_id, position LIMIT 10;
```

**Pipelines.** `Pipeline` chains transformers and a final estimator into one model (`steps`, as `{estimator, estimator_params}` specs); `FeatureUnion` concatenates transformers' outputs and `ColumnTransformer` applies them to column subsets:

```sql
WITH m AS (SELECT ml_fit_agg('Pipeline', y, [x1, x2, x3], '{"steps": [
         {"estimator": "StandardScaler", "estimator_params": {}},
         {"estimator": "Pca", "estimator_params": {"n_components": 2}},
         {"estimator": "RidgeRegressor", "estimator_params": {}}]}') AS model FROM train)
SELECT ml_r2(y, ml_predict(model, [x1, x2, x3])) FROM test, m;
```

**Kernel approximation.** `Nystroem`, `RbfSampler` and the random projections map features into a space where a linear model can fit non-linear structure. Stack them with `ml_transform`:

```sql
WITH f AS (SELECT ml_fit_agg('Nystroem', [a, b], '{"n_components": 100}') AS fm FROM train),
     m AS (SELECT fm, ml_fit_agg('LogisticRegressor', label, ml_transform(fm, [a, b])) AS model FROM train, f GROUP BY fm)
SELECT ml_predict_proba(model, ml_transform(fm, [a, b])) FROM test, m;
```

### Roadmap

| Topic | Issue |
|---|---|
| Everything else in anofox-ml 0.6 (tracker) | [#8](https://github.com/DataZooDE/anofox-ml/issues/8) |

## 🧮 Estimators

Pass the `kind` exactly as listed. `SELECT * FROM ml_estimators()` shows the defaults for each one.

| Category | Kinds |
|---|---|
| **Trees & Ensembles** | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `RandomForestClassifier`, `RandomForestRegressor`, `ExtraTreesClassifier`, `ExtraTreesRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`, `LgbmClassifier`, `LgbmRegressor`, `LgbmRanker`, `AdaBoostClassifier`, `AdaBoostRegressor`, `BaggingClassifier`, `BaggingRegressor`, `CatBoostClassifier`, `CatBoostRegressor`, `CatBoostMultiRegressor`, `PerpetualRegressor`, `PerpetualClassifier`, `ExtraTreeClassifier`, `ExtraTreeRegressor`, `BaggingEstimatorClassifier`, `BaggingEstimatorRegressor` (bagging over any kind) |
| **SVM** | `Svc`, `Svr`, `NuSvc`, `NuSvr`, `LinearSvc`, `LinearSvr` |
| **Neighbours** | `KnnClassifier`, `KnnRegressor`, `RadiusNeighborsClassifier`, `RadiusNeighborsRegressor`, `NearestCentroid`, `KNeighborsTransformer`, `NeighborhoodComponentsAnalysis` |
| **Naive Bayes** | `GaussianNB`, `BernoulliNB`, `MultinomialNB`, `ComplementNB`, `CategoricalNB` |
| **Discriminant Analysis** | `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |
| **Neural Networks** | `MlpClassifier`, `MlpRegressor`, `BernoulliRbm` |
| **Linear & GLM** | `OlsRegressor`, `RidgeRegressor`, `RidgeCrossValidated`, `LassoRegressor`, `LassoCrossValidated`, `ElasticNetRegressor`, `ElasticNetCrossValidated`, `Lars`, `OrthogonalMatchingPursuit`, `BayesianRidge`, `ARDRegression`, `HuberRegressor`, `QuantileRegressor`, `RansacRegressor`, `TheilSenRegressor`, `KernelRidge`, `LogisticRegressor`, `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`, `SgdClassifier`, `SgdRegressor`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor`, `RidgeClassifier`, `Perceptron`, `LogisticRegressionCV`, `OrthogonalMatchingPursuitCV`, `LassoLarsIC`, `LarsCV`, `LassoLarsCV`, `BinomialRegressor`, `IsotonicRegressor`, `MultiTaskLasso`, `MultiTaskElasticNet`, `MultiTaskLassoCV`, `MultiTaskElasticNetCV` |
| **Clustering** | `KMeans`, `MiniBatchKMeans`, `Dbscan`, `Hdbscan`, `Optics`, `MeanShift`, `Birch`, `AffinityPropagation`, `AgglomerativeClustering`, `SpectralClustering`, `GaussianMixture`, `BayesianGaussianMixture`, `BisectingKMeans` |
| **Outlier Detection** | `IsolationForest`, `LocalOutlierFactor`, `OneClassSvm`, `EllipticEnvelope` |
| **Covariance & Density** | `EmpiricalCovariance`, `ShrunkCovariance`, `LedoitWolf`, `Oas`, `MinCovDet`, `KernelDensity` |
| **Decomposition & Manifold** | `Pca`, `IncrementalPca`, `KernelPca`, `TruncatedSvd`, `FastIca`, `Nmf`, `ClassicalMds`, `Isomap`, `LocallyLinearEmbedding`, `TSne`, `FactorAnalysis`, `FeatureAgglomeration`, `Nystroem`, `RbfSampler`, `GaussianRandomProjection`, `SparseRandomProjection`, `SparsePca`, `MiniBatchNmf`, `LatentDirichletAllocation`, `SpectralEmbedding`, `RandomTreesEmbedding`, `PlsRegression`, `PlsCanonical`, `PlsSvd`, `Cca` |
| **Preprocessing** | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `Binarizer`, `PowerTransformer`, `QuantileTransformer`, `KBinsDiscretizer`, `PolynomialFeatures`, `SplineTransformer`, `RcsTransformer`, `TargetEncoder`, `SimpleImputer`, `KnnImputer`, `IterativeImputer`, `MissingIndicator`, `VarianceThreshold`, `SelectKBest`, `GenericUnivariateSelect`, `MutualInformationSelector`, `KernelCenterer`, `TfidfTransformer`, `CountVectorizer`, `TfidfVectorizer` (VARCHAR input), `LabelEncoder`, `OrdinalEncoder` (numeric categories; `VARCHAR` columns see [Categorical Features](#categorical-features)) |
| **Meta-estimators** | `OneVsRestClassifier`, `OneVsOneClassifier`, `FixedThresholdClassifier`, `TunedThresholdClassifierCV`, `SelfTrainingClassifier`; they take a base `estimator` kind and `estimator_params`, `VotingClassifier`, `VotingRegressor`, `StackingClassifier`, `StackingRegressor`, `CalibratedClassifierCV`, `TransformedTargetRegressor`, `MultiOutputRegressor`, `RegressorChain`, `Rfe`, `Rfecv`, `SequentialFeatureSelector`, `SelectFromModel`, and the compositions `Pipeline`, `FeatureUnion`, `ColumnTransformer` |
| **Conformal** | `SplitConformalRegressor`, `CrossConformalRegressor`, `NormalizedConformalRegressor`, `ConformalizedQuantileRegressor`, `MondrianConformalRegressor`, `SplitConformalClassifier`, `MondrianConformalClassifier`; intervals and sets through `ml_predict_interval` / `ml_predict_set`, see [Conformal Prediction](#conformal-prediction) |
| **Baselines & Semi-supervised** | `DummyClassifier`, `DummyRegressor`, `LabelPropagation`, `LabelSpreading` (a NULL target marks an unlabelled row, see Behaviour Notes) |
| **Gaussian Processes** | `GaussianProcessRegressor`, `GaussianProcessClassifier` |

Linear models and GLMs **with statistical inference** (standard errors, tests, diagnostics) live in [anofox-statistics](https://github.com/DataZooDE/anofox-statistics). The linear models here are for prediction; `OlsRegressor` / `RidgeRegressor` with `inference: true` add a coefficient table and prediction intervals. `IsotonicRegressor` takes exactly one feature; `PlsRegression` takes one target `y`.

## 📝 Behaviour Notes

- **NULLs:** rows with a NULL target, weight, feature list or NULL value inside the feature list are skipped when fitting, and predict NULL. The per-row aggregates return a NULL entry for such rows.
- **Semi-supervised kinds** (`LabelPropagation`, `LabelSpreading`, `SelfTrainingClassifier`): a row with a NULL target is an *unlabelled* row and takes part in the fit; only NULL features skip a row. `-1` is never a class of these kinds (the library's unlabelled marker), and `classes` / `labels` list only the real classes. `ml_tune_agg` still skips unlabelled rows.
- **Weights:** kinds without weighted fitting reject a `weight` with an error naming the kind.
- **Row order:** `ml_cluster_agg`, `ml_outlier_score_agg` and `ml_fit_transform_agg` return one entry per input row in input order. That order is only well defined with `ORDER BY` inside the aggregate.
- **Classes:** for models with `predict_proba`, the sorted distinct training targets are stored as `classes`. With `VARCHAR` targets, the sorted distinct labels are stored as `labels`, the model learns their index, and `classes` / `ml_predict_proba` columns follow the label order.
- **Feature count:** predicting with a feature list of the wrong length is an error.
- **Stored models across versions:** a model BLOB keeps predicting the same after an upgrade of the extension. BLOBs written with anofox-ml 0.2, 0.4 and 0.5 are tested against the outputs they gave then (`test/data/blobs`). Refitting after an upgrade can give different numbers when a kind's training changed: anofox-ml 0.4 changed LGBM, HGB early stopping, SGD and SelectKBest ties; 0.5 changed LightGBM's bagging (`subsample` needs `subsample_freq > 0`) and some CatBoost defaults.
- **Memory:** a fitting aggregate holds its group's training set in memory. `SET anofox_ml_max_fit_values = n` caps it per group (rows × features, default 100 million ≈ 800 MB; `0` = unlimited). Past the cap the query fails with a hint; sample large groups with `USING SAMPLE` before fitting.
- **Parallelism:** fits run in one shared thread pool sized to DuckDB's `threads`. DuckDB spreads groups over its workers, and estimators with internal parallelism (forests, bagging, isolation forests, …) use the pool too, so a single big model uses every core while many groups still share the same threads. `SET anofox_ml_threads = n` caps the threads of the fits (`0` = DuckDB's `threads`, `1` = sequential). Predictions run single-threaded per chunk, with DuckDB parallelising across chunks. Seeded models are identical whatever the thread count.
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
- Rust ≥ 1.88 (the PerpetualBooster kinds need it)

### Build and Test

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-ml.git
cd anofox-ml

GEN=ninja make release            # build/release/duckdb with the extension loaded
make test                         # SQL logic tests in test/sql
cargo test                        # Rust FFI tests
cargo fmt --all                   # format
cargo clippy --all-targets        # lint
make check_skills                 # run the SQL examples of this README and the skills
```

### Layout

| Path | Contents |
|---|---|
| `src/` | C++ extension: thin DuckDB wrappers around the C ABI |
| `crates/anofox-ml-ffi/` | Rust `staticlib`: generic C ABI over [`anofox-ml-registry`](https://crates.io/crates/anofox-ml-registry) |
| `test/sql/` | sqllogictests |
| `plugins/anofox-ml/` | Claude Code plugin (skills); `scripts/check_skill_snippets.py` runs their SQL |

The estimators come from [anofox-ml](https://github.com/sipemu/anofox-ml) 0.6 on crates.io (`anofox-ml-registry`, `-core`, `-metrics`, `-conformal`, `-text`).

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
