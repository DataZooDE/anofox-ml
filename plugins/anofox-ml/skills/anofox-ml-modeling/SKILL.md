---
name: anofox-ml-modeling
description: >
  Fitting and predicting with the anofox_ml DuckDB extension: ml_fit_agg over
  any of its 188 estimator kinds (trees, boosting, linear / GLM, SVM, kNN,
  naive Bayes, MLP, Gaussian processes), per GROUP BY group, models stored as
  BLOBs, ml_predict / ml_predict_proba / ml_predict_label and the other model
  functions, JSON params, string labels, sample weights, ml_estimators,
  model inspection (ml_model_info, ml_coefficients, ml_feature_importances,
  ml_model_fitted, ml_kneighbors), continuing a stored model
  (ml_warm_start_agg, ml_partial_fit_agg, ml_staged_predict,
  ml_fit_with_validation_agg) and the ml_fit_predict_by table macros. Use
  for "train / score a model in SQL", "one model per store / segment", or
  any question about these functions.
version: 0.2.0
user-invocable: false
---

# anofox_ml — fitting, predicting, models as data

**Extension:** `anofox_ml` v0.2.0 (anofox-ml 0.5) | **DuckDB:** v1.5.6 / v2.0.0 | `INSTALL anofox_ml FROM community; LOAD anofox_ml;`

The model is a value. `ml_fit_agg` is an aggregate that fits one model per group and returns a `BLOB`; every later query predicts by passing that BLOB and a feature list to a scalar function. Features are always one `DOUBLE[]` list, `[x1, x2, …]`, in the same order at fit and predict time.

## Fit

```sql
-- synopsis
ml_fit_agg(kind, y, [x1, x2, …] [, params JSON [, weight]])   -- supervised
ml_fit_agg(kind, [x1, x2, …] [, params JSON [, weight]])      -- unsupervised (scalers, PCA, clustering, …)
```

- `kind` is case-sensitive (`'HistGradientBoostingRegressor'`, `'Svc'`, `'KnnClassifier'`, `'Pca'`). Find it, its defaults and its methods in `ml_estimators()`.
- `params` is a JSON object merged into the defaults; a misspelled name is an error listing the valid ones.
- `y` is `DOUBLE`, or `VARCHAR` labels for a classifier.
- Rows with a NULL target, weight or feature are skipped (semi-supervised kinds keep NULL-target rows as unlabelled).

```sql
-- one model per store, kept in a table
CREATE TABLE models AS
SELECT store, ml_fit_agg('HistGradientBoostingRegressor', units, [price, promo, dow],
                         '{"n_estimators": 200, "learning_rate": 0.05}') AS model
FROM train GROUP BY store;

SELECT store, ml_rmse(units, ml_predict(model, [price, promo, dow])) AS rmse
FROM test JOIN models USING (store) GROUP BY store ORDER BY store;
```

```sql
-- what is there: probabilistic classifiers, and one kind's defaults
SELECT kind FROM ml_estimators() WHERE list_contains(capabilities, 'predict_proba') ORDER BY kind LIMIT 5;
SELECT default_params FROM ml_estimators() WHERE kind = 'RandomForestClassifier';
```

## Predict

| Function | Returns |
|---|---|
| `ml_predict(model, x)` | `DOUBLE`: value, class, or the label index for string labels |
| `ml_predict_label(model, x)` | `VARCHAR` class label |
| `ml_predict_proba(model, x)` / `ml_predict_log_proba` | `DOUBLE[]` in the order of `ml_model_info(model).classes` |
| `ml_decision_function(model, x)` | `DOUBLE[]` margins |
| `ml_transform(model, x)` / `ml_inverse_transform` | `DOUBLE[]` features (scalers, PCA, encoders, selectors) |
| `ml_score_samples(model, x)` | per-row scores (densities, outlier scores) |
| `ml_predict_std(model, x)` | predictive std of `GaussianProcessRegressor` |
| `ml_predict_interval(model, x, alpha)` / `ml_predict_set` | intervals and sets: see the evaluation skill |

A model rejects a method it doesn't have with an error naming the kinds that do; `ml_model_info(model).capabilities` lists what it supports. A feature list of the wrong length is an error.

```sql
-- string labels in, labels and probabilities out
WITH m AS (SELECT ml_fit_agg('RandomForestClassifier', churn, [tenure, calls, spend], '{"seed": 0}') AS model FROM customers)
SELECT ml_model_info(model).labels AS labels,
       ml_predict_label(model, [3.0, 5.0, 2.0]) AS label,
       ml_predict_proba(model, [3.0, 5.0, 2.0]) AS proba
FROM m;
```

```sql
-- sample weights (5th argument; params must then be given, '{}' for defaults)
SELECT ml_coefficients(ml_fit_agg('OlsRegressor', units, [price, promo], '{}', 1.0 + promo)) FROM train;
```

## Inspect a model

| Function | Gives |
|---|---|
| `ml_model_info(model)` | `STRUCT`: `kind`, `n_features`, `classes`, `labels`, `capabilities`, `params`, `n_iter`, `inertia`, `oob_score`, versions |
| `ml_model_kind(model)` | the kind, without decoding |
| `ml_coefficients(model)` | `STRUCT(coef DOUBLE[][], intercept DOUBLE[])` of linear models |
| `ml_feature_importances(model)` | impurity importances of tree / boosting / bagging kinds, sum 1 |
| `ml_cluster_centers(model)`, `ml_explained_variance_ratio(model)` | centroids; PCA-like variance ratios |
| `ml_model_fitted(model)` | the whole fitted state as JSON (arrays as `{"v","dim","data"}`) |
| `ml_kneighbors(model, x [, k])` | `STRUCT(indices, distances)` of a kNN model's nearest training rows, 0-based in fit input order (fit with `ORDER BY` to map back) |

For model-agnostic importance use `ml_permutation_importance_agg` (evaluation skill). For p-values and standard errors use `OlsRegressor` / `RidgeRegressor` with `{"inference": true}` (coefficient table under `summary` in `ml_model_fitted`), or the anofox_statistics extension.

```sql
WITH m AS (SELECT ml_fit_agg('KnnRegressor', units, [price, dow], '{"n_neighbors": 3}' ORDER BY id) AS model FROM train)
SELECT ml_feature_importances(ml_fit_agg('RandomForestRegressor', units, [price, promo, dow], '{"n_estimators": 20, "seed": 0}')) AS fi,
       (SELECT ml_kneighbors(model, [1.0, 3.0]) FROM m) AS nn
FROM train;
```

## Continue a stored model

| Function | For |
|---|---|
| `ml_warm_start_agg(model, y, x [, params])` | more boosting stages (GradientBoosting, HGB, LGBM: raise `n_estimators`) or MLP epochs. 100 + 100 trees on the same rows = a 200-tree fit |
| `ml_partial_fit_agg(model, [y,] x [, params])` | one more batch: naive Bayes, SGD, PassiveAggressive, MLP; without `y` StandardScaler / MinMaxScaler / MaxAbsScaler / IncrementalPca. Classes are fixed by the first fit |
| `ml_staged_predict(model, x)` | boosting: prediction after every stage (`DOUBLE[]`), for learning curves |
| `ml_fit_with_validation_agg(kind, y, x, is_validation [, params [, weight]])` | early stopping on rows you mark as validation (HGB with `"early_stopping": "Enabled"`) |

```sql
CREATE TABLE nb AS SELECT ml_fit_agg('GaussianNB', churn, [tenure, calls]) AS model FROM customers WHERE id < 500;
-- fold in the next batch; the second argument is the stored model
SELECT ml_predict_label(m2, [3.0, 5.0]) FROM (
  SELECT ml_partial_fit_agg(nb.model, churn, [tenure, calls]) AS m2 FROM customers, nb WHERE id >= 500 GROUP BY nb.model);
```

## One-shot table macros

Pass the table name as a string; columns unquoted; `1` as `group_col` for one model over everything. Rows with a NULL target are predicted but not trained on (append future rows with `y = NULL` to forecast).

| Macro | Adds |
|---|---|
| `ml_fit_predict_by(source, group_col, y_col, x_cols, kind [, params := …])` | `prediction` |
| `ml_fit_predict_proba_by(…same…)` | `prediction`, `label`, `proba`, `classes`, `labels` |

```sql
SELECT store, ml_r2(units, prediction) AS in_sample_r2
FROM ml_fit_predict_by('train', store, units, [price, promo, dow], 'RandomForestRegressor', params := '{"n_estimators": 30}')
GROUP BY store ORDER BY store;
```

## Gotchas

- Per-row aggregates (`ml_cluster_agg`, `ml_fit_transform_agg`, …) and `ml_kneighbors` indices follow input order: put `ORDER BY id` inside the aggregate.
- A fitting aggregate holds its group in memory; `SET anofox_ml_max_fit_values = n` caps rows × features per group (default 100 million, `0` = off). Sample huge groups with `USING SAMPLE`.
- BLOBs from older versions keep predicting the same after an upgrade; refitting may differ where a kind's training changed.
- DuckDB-Wasm (browser): the same SQL works, single-threaded and within the tab's memory; `PerpetualRegressor` / `PerpetualClassifier` are not in the wasm builds (use `HistGradientBoosting*`).
- Categorical and text inputs, clustering and decomposition: the features skill. Metrics, CV, tuning, conformal intervals: the evaluation skill. Ensembles, offsets, multi-target, drift: the advanced skill.
