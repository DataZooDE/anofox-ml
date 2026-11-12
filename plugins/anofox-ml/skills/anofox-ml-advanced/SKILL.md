---
name: anofox-ml-advanced
description: >
  Composite and specialised estimators of the anofox_ml DuckDB extension:
  nested {estimator, estimator_params} specs for ensembles (VotingRegressor /
  VotingClassifier, StackingRegressor / StackingClassifier, BaggingEstimator*),
  one-vs-rest, decision thresholds, CalibratedClassifierCV,
  TransformedTargetRegressor, semi-supervised fits; GLM exposure / offsets
  (ml_fit_with_offset_agg), several targets at once (MultiOutputRegressor,
  RegressorChain, ml_predict_targets), Cca / PLS (ml_transform_y), Gaussian
  process uncertainty (ml_predict_std), CatBoost categorical features, and
  production monitoring: multivariate drift (ml_domain_classifier_drift_by)
  and label-free performance estimation (ml_dle_estimate_by). Use when one
  plain ml_fit_agg kind is not enough, for learning to rank
  (ml_fit_with_group_agg, LgbmRanker) or pipelines, or to monitor a
  deployed model in SQL.
version: 0.2.0
user-invocable: false
---

# anofox_ml — composite estimators, special targets, monitoring

**Extension:** `anofox_ml` v0.2.0 | Fitting and predicting basics are in the modeling skill.

Every kind here still fits with `ml_fit_agg` and predicts with the usual functions; the composition lives in the `params` JSON. A nested model is a spec `{"estimator": "<kind>", "estimator_params": {…}}`, and `SELECT default_params FROM ml_estimators() WHERE kind = '…'` shows every option of a wrapper.

## Wrappers and ensembles

| Kind | What the params say |
|---|---|
| `OneVsRestClassifier`, `OneVsOneClassifier`, `BaggingEstimatorClassifier` / `BaggingEstimatorRegressor` | one nested `estimator` |
| `FixedThresholdClassifier` | `estimator`, `threshold` on the positive-class probability |
| `TunedThresholdClassifierCV` | threshold tuned by CV for `scoring`: `BalancedAccuracy` (default), `Accuracy`, `F1`, `Precision`, `Recall` (binary) |
| `VotingRegressor` / `VotingClassifier` | `estimators: [spec, …]`, optional `weights`; classifier `voting` `"hard"` / `"soft"` |
| `StackingRegressor` / `StackingClassifier` | `estimators: [spec, …]`, `final_estimator: spec`, `cv`, `passthrough`, `stack_method` |
| `CalibratedClassifierCV` | `estimator` (default `LinearSvc`), `method` `"sigmoid"` / `"isotonic"`, `cv`, `ensemble`: probabilities for margin classifiers |
| `TransformedTargetRegressor` | `regressor: spec` fitted on `func(y)` (`"log"`, `"log1p"`, `"sqrt"`) or on a `transformer` kind's output; predictions mapped back |
| `SelfTrainingClassifier`, `LabelPropagation`, `LabelSpreading` | semi-supervised: rows with a NULL target are the unlabelled rows of the fit |

```sql
SELECT ml_predict_proba(model, [3.0, 5.0]) AS proba FROM (
  SELECT ml_fit_agg('VotingClassifier', churn, [tenure, calls], '{"voting": "soft", "weights": [1, 2],
         "estimators": [{"estimator": "GaussianNB", "estimator_params": {}},
                        {"estimator": "HistGradientBoostingClassifier", "estimator_params": {"n_estimators": 50}}]}') AS model
  FROM customers);
```

```sql
SELECT ml_r2(units, ml_predict(model, [price, promo, dow])) AS test_r2 FROM test, (
  SELECT ml_fit_agg('StackingRegressor', units, [price, promo, dow], '{"cv": 3,
         "estimators": [{"estimator": "RidgeRegressor", "estimator_params": {}},
                        {"estimator": "RandomForestRegressor", "estimator_params": {"n_estimators": 20}}],
         "final_estimator": {"estimator": "OlsRegressor", "estimator_params": {}}}') AS model FROM train);
```

```sql
-- log-linear model for a skewed positive target
SELECT ml_predict(model, [1.0, 0.0, 3.0]) FROM (
  SELECT ml_fit_agg('TransformedTargetRegressor', revenue, [price, promo, dow],
         '{"func": "log", "regressor": {"estimator": "RidgeRegressor", "estimator_params": {}}}') AS model FROM train);
```

## Special targets

- **Rates with exposure (GLM offsets).** `ml_fit_with_offset_agg(kind, y, [x…], offset [, params [, weight]])` for `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor`, `BinomialRegressor` (proportions `y`, trials as `weight`). The offset is on the link scale, e.g. `ln(exposure)`. `ml_predict(model, x, offset)` gives the expected value at that exposure; `ml_predict(model, x)` the value at offset 0 (the rate).
- **Several targets.** `MultiOutputRegressor` (nested `estimator` per target), `RegressorChain` (`order`), `MultiTaskLasso` / `MultiTaskElasticNet` (+ CV kinds) fit unsupervised on `list_concat(x, y)` with `y_columns` = number of targets; predict with X alone through `ml_predict_targets(model, x)` → `DOUBLE[]`.
- **Two blocks.** `Cca`, `PlsCanonical`, `PlsSvd` fit on `[X | Y]` the same way; `ml_transform` scores X, `ml_transform_y(model, y)` scores Y. `PlsRegression` is an ordinary one-target regressor.
- **Uncertainty.** `GaussianProcessRegressor` (default kernel `1.0 * RBF(1.0)`): `ml_predict_std(model, x)` next to `ml_predict`.
- **Learning to rank.** `ml_fit_with_group_agg(kind, relevance, [x…], query_id [, params [, weight]])` for `LgbmRanker` (LambdaRank) or the ranking objectives of `LgbmRegressor` / `CatBoostRegressor`; `ml_predict` gives scores, rank within a query by `row_number() OVER (PARTITION BY query_id ORDER BY score DESC)`.
- **Pipelines.** `Pipeline` (`steps`: specs, transformers then an estimator), `FeatureUnion` (`transformer_list`), `ColumnTransformer` (`transformers`, `remainder`) and `SelectFromModel` are kinds like any other: one BLOB, `ml_predict` / `ml_transform` on the raw features.
- **CatBoost categories.** `CatBoostRegressor` / `CatBoostClassifier` take `cat_features` (0-based positions in the feature list) holding integer codes, e.g. `dense_rank() OVER (ORDER BY city) - 1`.

```sql
WITH p AS (SELECT *, 1 + id % 4 AS years, (hash(id, 'k') % 3)::DOUBLE AS claims FROM customers),
     m AS (SELECT ml_fit_with_offset_agg('PoissonRegressor', claims, [tenure, calls], ln(years), '{"alpha": 0}') AS model FROM p)
SELECT ml_predict(model, [24.0, 2.0], ln(3)) AS expected_in_3_years, ml_predict(model, [24.0, 2.0]) AS per_year FROM m;
```

```sql
WITH m AS (SELECT ml_fit_agg('MultiOutputRegressor', [price, promo, dow, units, revenue],
                  '{"estimator": "RidgeRegressor", "y_columns": 2}') AS model FROM train)
SELECT ml_predict_targets(model, [1.0, 0.0, 3.0]) AS units_and_revenue FROM m;
```

```sql
WITH m AS (SELECT ml_fit_agg('GaussianProcessRegressor', units, [dow]) AS gp FROM train WHERE store = 'store_0' AND day < 100)
SELECT d AS dow, ml_predict(gp, [d]) AS mean, ml_predict_std(gp, [d]) AS sd FROM m, (SELECT unnest([0.0, 3.0, 6.0, 20.0]) AS d);
```

## Monitoring a deployed model

| Macro | Answers |
|---|---|
| `ml_domain_classifier_drift_by(source, chunk_col, id_col, x_cols, reference [, kind, params, folds := 5, threshold := 0.55, seed])` | Did the inputs shift? A classifier tells each chunk's rows from the `reference` chunk's; out-of-fold ROC AUC ≈ 0.5 means no drift. Returns `chunk, n_reference, n_rows, auc, threshold, drifted` |
| `ml_dle_estimate_by(reference, analysis, chunk_col, y_col, pred_col, x_cols [, kind, params])` | How large is the error while labels are not in yet? Direct loss estimation (NannyML DLE): models of the absolute / squared error learned on labelled `reference` rows. Returns `chunk, n_rows, estimated_mae, mae_sampling_error, estimated_rmse, rmse_sampling_error` |

`reference` / `analysis` / `source` are table names as strings. To see which features drift, fit a classifier on reference vs. the drifted chunk and read `ml_feature_importances`.

```sql
CREATE TABLE weekly AS SELECT id, 'w' || (day // 200) AS week, price + CASE WHEN day >= 800 THEN 0.5 ELSE 0 END AS price, dow FROM sales;
SELECT chunk, round(auc, 2) AS auc, drifted FROM ml_domain_classifier_drift_by('weekly', week, id, [price, dow], 'w0') ORDER BY chunk;
```

```sql
CREATE TABLE ref AS SELECT *, ml_predict(m, [price, promo, dow]) AS forecast
FROM train, (SELECT ml_fit_agg('RidgeRegressor', units, [price, promo, dow]) AS m FROM train WHERE day < 400) WHERE day >= 400;
CREATE TABLE prod AS SELECT *, ml_predict(m, [price, promo, dow]) AS forecast, day // 50 AS chunk
FROM test, (SELECT ml_fit_agg('RidgeRegressor', units, [price, promo, dow]) AS m FROM train WHERE day < 400);
SELECT chunk, round(estimated_mae, 2) AS est_mae FROM ml_dle_estimate_by('ref', 'prod', chunk, units, forecast, [price, promo, dow]) ORDER BY chunk;
```
