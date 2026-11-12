---
name: anofox-ml-evaluation
description: >
  Evaluating, validating and tuning models with the anofox_ml DuckDB
  extension: metric aggregates (ml_rmse, ml_r2, ml_accuracy, ml_f1,
  ml_roc_auc, ml_log_loss, …, matching scikit-learn), cross-validation
  folds (ml_cv_splits_agg, ml_purged_cv_splits_agg,
  ml_spatial_cv_splits_agg, ml_cross_val_predict_by), hyperparameter search
  (ml_tune_agg: grid, random, successive halving, TPE), permutation
  importance and recursive feature elimination (ml_rfe_agg), and
  uncertainty: conformal prediction intervals / sets (ml_conformalize_agg,
  ml_predict_interval, ml_predict_set, conformal kinds) with coverage
  checks. Use for "how good is this model", "cross-validate", "tune
  hyperparameters", "which features matter" or "prediction intervals" in
  SQL.
version: 0.3.0
user-invocable: false
---

# anofox_ml — metrics, cross-validation, tuning, intervals

**Extension:** `anofox_ml` v0.3.0 | Fitting and predicting basics are in the modeling skill.

Evaluate on rows the model did not train on: a hold-out table, CV folds, or `ml_cross_val_predict_by`. In-sample scores of flexible models (forests, boosting, kNN) are near perfect and say nothing.

## Metrics

Aggregates over `(y_true, y_pred)`: they score any prediction column, per `GROUP BY` group. NULL rows are skipped; an empty group gives NULL.

| Task | Aggregates |
|---|---|
| Regression `(y, y_pred)` | `ml_r2`, `ml_mse`, `ml_rmse`, `ml_mae`, `ml_mape`, `ml_median_absolute_error`, `ml_max_error`, `ml_explained_variance`, `ml_mean_squared_log_error` |
| Labels `(y, y_pred [, options [, weight]])` | `ml_accuracy`, `ml_precision`, `ml_recall`, `ml_f1`, `ml_fbeta`; options `average` (`binary` default, `macro`, `micro`, `weighted`), `pos_label`, `zero_division`, `labels`, `beta`. `DOUBLE` or `VARCHAR` labels |
| Labels `(y, y_pred)` | `ml_balanced_accuracy`, `ml_matthews_corrcoef` (binary), `ml_cohen_kappa` |
| Scores / probabilities | `ml_roc_auc(y, score \| proba[])`, `ml_log_loss(y, proba[])`, `ml_brier_score(y, p)`, `ml_average_precision(y, score)` |
| Clustering | `ml_adjusted_rand`, `ml_normalized_mutual_info`, `ml_silhouette(x DOUBLE[], label)` |
| Intervals / quantiles | `ml_coverage(y, lower, upper)`, `ml_interval_score(y, lower, upper, alpha)`, `ml_mean_pinball_loss(y, q, alpha)` |

For a binary model, `ml_predict_proba(model, x)[2]` is the probability of the second of `ml_model_info(model).classes` (the positive class for 0/1 targets).

```sql
WITH m AS (SELECT ml_fit_agg('LogisticRegressor', churned, [tenure, calls]) AS model FROM customers WHERE id < 700)
SELECT ml_roc_auc(churned, ml_predict_proba(model, [tenure, calls])[2]) AS auc,
       ml_log_loss(churned, ml_predict_proba(model, [tenure, calls])) AS log_loss,
       ml_f1(churned, ml_predict(model, [tenure, calls])) AS f1
FROM customers, m WHERE id >= 700;
```

## Cross-validation

`ml_cross_val_predict_by` is the one-call route: out-of-fold predictions for every labelled row, plus its `split`.

```sql
SELECT store, ml_rmse(units, prediction) AS cv_rmse
FROM ml_cross_val_predict_by('train', store, id, units, [price, promo, dow], 'RandomForestRegressor',
                             params := '{"n_estimators": 30}', options := '{"n_splits": 5, "shuffle": true}')
GROUP BY store ORDER BY store;
```

For custom loops, `ml_cv_splits_agg(id, method, options [, y [, groups]] ORDER BY …)` returns `STRUCT(id, split, role)` rows (`role` = `'train'` / `'test'`); `unnest(…, recursive := true)` makes a fold table to join back. The `ORDER BY` fixes the row order, which is the time order for time-series methods.

| Method | Needs | Options |
|---|---|---|
| `kfold`, `repeated_kfold`, `shuffle_split`, `leave_one_out`, `leave_p_out` | — | `n_splits`, `shuffle`, `seed`, `n_repeats`, `test_size`, `p` |
| `time_series` | — | `n_splits`, `test_size`, `gap`, `max_train_size`, `min_train_size` |
| `stratified_kfold`, `repeated_stratified_kfold`, `stratified_shuffle_split`, `stratified_kfold_continuous` | `y` | `n_splits`, `n_repeats`, `test_size`, `n_bins`, `seed` |
| `group_kfold`, `leave_one_group_out`, `leave_p_groups_out`, `group_shuffle_split`, `group_time_series` | `groups` (pass `NULL` for `y`) | `n_splits`, `shuffle`, `seed`, `p`, `test_size` |
| `stratified_group_kfold` | `y`, `groups` | `n_splits`, `shuffle`, `seed` |

- Labels spanning time (forward returns, churn windows): `ml_purged_cv_splits_agg(id, t_start, t_end, method, options ORDER BY t_start)` with `purged_kfold` or `combinatorial_purged` (`embargo`, `n_groups`, `n_test_groups`; test rows carry a backtest `path`).
- Spatial data: `ml_spatial_cv_splits_agg(id, coords DOUBLE[], method, options ORDER BY id)` with `spatial_block_kfold` (`block_size`), `kfold` or `buffered_loo` (`buffer`); `metric` `"haversine"` for `[lat, lon]`.

```sql
WITH folds AS (SELECT unnest(ml_cv_splits_agg(id, 'time_series', '{"n_splits": 4}' ORDER BY day, id), recursive := true) FROM train WHERE store = 'store_0'),
     f AS (SELECT * FROM folds JOIN train USING (id)),
     m AS (SELECT split, ml_fit_agg('RidgeRegressor', units, [price, promo, dow]) AS model FROM f WHERE role = 'train' GROUP BY split)
SELECT split, ml_rmse(units, ml_predict(model, [price, promo, dow])) AS rmse
FROM f JOIN m USING (split) WHERE role = 'test' GROUP BY split ORDER BY split;
```

## Hyperparameter search

`ml_tune_agg(kind, y, [x…], search_space [, options])` → `STRUCT(best_params, best_score, model, trials)`. `model` is the best candidate refitted on all rows; `trials` is JSON with every candidate's mean / std CV score or its error. One search per group.

- **Search space** JSON, one entry per parameter: a list of values, `{"int": [lo, hi]}`, `{"float": [lo, hi]}` (either with `"log": true`), `{"categorical": […]}`.
- **Options:** `search` = `"random"` (default), `"grid"` (lists only), `"halving"` / `"halving_grid"` (successive halving, `factor` 3), `"tpe"` (Bayesian, Optuna-style: `n_startup_trials` 10, `n_ei_candidates` 24); `n_trials` (20), `cv` (5; stratified for classifiers), `metric` (any metric aggregate name; losses are minimised; default `accuracy` / `r2`), `metric_options`, `params` (fixed for all), `seed`.
- Seeded and thread-count independent. Cost ≈ `n_trials × cv` fits per group; halving cuts that, TPE spends it more wisely.

```sql
SELECT store, t.best_params, round(t.best_score, 3) AS cv_rmse, ml_predict(t.model, [1.0, 0.0, 3.0]) AS pred
FROM (SELECT store, ml_tune_agg('HistGradientBoostingRegressor', units, [price, promo, dow],
        '{"max_depth": {"int": [2, 8]}, "learning_rate": {"float": [0.02, 0.3], "log": true}}',
        '{"search": "tpe", "n_trials": 12, "n_startup_trials": 5, "metric": "rmse", "cv": 3, "params": {"n_estimators": 50}}') AS t
      FROM train GROUP BY store) ORDER BY store;
```

## Which features matter

| Function | Use |
|---|---|
| `ml_permutation_importance_agg(model, y, [x…] [, options])` | any model, over evaluation rows: `STRUCT(feature (1-based), importance_mean, importance_std)[]`; options `metric`, `n_repeats` (5), `seed` |
| `ml_feature_importances(model)` | impurity importances of tree kinds (training data, biased toward high-cardinality features) |
| `ml_rfe_agg(kind, y, [x…] [, params [, options]])` | recursive elimination, scikit-learn `RFE`: `STRUCT(support, ranking, model)`; options `n_features_to_select`, `step` |
| `Rfecv`, `SequentialFeatureSelector` kinds | selection by CV score, through `ml_fit_agg` + `ml_transform` |

```sql
WITH m AS (SELECT ml_fit_agg('RandomForestRegressor', units, [price, promo, dow, noise], '{"n_estimators": 30, "seed": 0}') AS model FROM train)
SELECT unnest(ml_permutation_importance_agg(model, units, [price, promo, dow, noise], '{"n_repeats": 3}'), recursive := true)
FROM test, m GROUP BY model;
```

## Prediction intervals and sets

Coverage ≥ `1 − alpha` holds when calibration and new rows are exchangeable. Check it on held-out rows with `ml_coverage` (intervals) or `ml_set_coverage` / `ml_mean_set_size` (sets).

- **Any fitted model:** calibrate on held-out rows with `ml_conformalize_agg(model, y, [x…])`, then `ml_predict_interval(model, x, alpha)` → `STRUCT(lower, upper)` (regressors) or `ml_predict_set(model, x, alpha)` → `VARCHAR[]` (classifiers), at any `alpha`.
- **Conformal kinds** calibrate themselves at fit time; base model as `estimator` + `estimator_params`: `SplitConformalRegressor`, `CrossConformalRegressor` (CV+ / jackknife+), `NormalizedConformalRegressor` (adaptive widths), `ConformalizedQuantileRegressor` (CQR, `alpha` fixed at fit), `MondrianConformalRegressor` / `MondrianConformalClassifier` (coverage per group id in feature `group_column`), `SplitConformalClassifier` (LAC / APS / RAPS).
- **Classical:** `OlsRegressor` / `RidgeRegressor` with `{"inference": true}` give Gaussian prediction intervals through `ml_predict_interval`, no calibration set.
- `ml_conformal_margin(y, y_pred, alpha)` is the model-free conformal half-width of any prediction column.

```sql
CREATE TABLE cal AS
SELECT ml_conformalize_agg(m, units, [price, promo, dow]) AS m
FROM train, (SELECT ml_fit_agg('HistGradientBoostingRegressor', units, [price, promo, dow]) AS m FROM train WHERE day < 600)
WHERE day >= 600 GROUP BY ALL;
SELECT ml_coverage(units, pi.lower, pi.upper) AS coverage, avg(pi.upper - pi.lower) AS width
FROM (SELECT units, ml_predict_interval(m, [price, promo, dow], 0.1) AS pi FROM test, cal);
```

```sql
WITH m AS (SELECT ml_fit_agg('SplitConformalClassifier', churn, [tenure, calls],
                  '{"estimator": "LogisticRegressor", "estimator_params": {}}') AS model FROM customers WHERE id < 700)
SELECT ml_set_coverage(churn, s) AS coverage, ml_mean_set_size(s) AS mean_size
FROM (SELECT churn, ml_predict_set(model, [tenure, calls], 0.1) AS s FROM customers, m WHERE id >= 700);
```
