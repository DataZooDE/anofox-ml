# Changelog

## [0.1.0] - 2026-11-08

First release. It is built for **DuckDB v1.5.6** and bundles [anofox-ml](https://github.com/sipemu/anofox-ml) **0.4** from crates.io. The binaries below are unsigned; see [Installation](https://github.com/DataZooDE/anofox-ml#installation). The API may still change before the community-extension release (#4).

### Estimators
- 144 scikit-learn-compatible estimators, selected by `kind` (`SELECT * FROM ml_estimators()`). They cover trees and boosting (including LightGBM- and CatBoost-style boosting), SVMs, neighbours, naive Bayes, discriminant analysis, MLPs, linear models and GLMs, clustering, outlier detection, decomposition and manifold learning, covariance and density, preprocessing, meta-estimators (one-vs-rest, one-vs-one, threshold tuning, self-training), conformal models and baselines.

### SQL surface
- **Fit and predict:** `ml_fit_agg` fits one model per `GROUP BY` group and returns it as a `BLOB`. Then `ml_predict`, `ml_predict_label`, `ml_predict_proba`, `ml_predict_log_proba`, `ml_decision_function`, `ml_transform` / `ml_inverse_transform`, `ml_score_samples`. JSON parameters, sample weights, and string labels or categorical features.
- **Per-row aggregates:** `ml_cluster_agg`, `ml_outlier_score_agg`, `ml_fit_transform_agg`, and the fit-predict table macros (`ml_fit_predict_by`, `ml_cluster_by`, `ml_outlier_score_by`).
- **Model inspection:** `ml_model_info`, `ml_model_kind`, `ml_model_fitted`, `ml_coefficients`, `ml_cluster_centers`, `ml_explained_variance_ratio`, `ml_permutation_importance_agg`.
- **Evaluation:** metric aggregates (`ml_r2`, `ml_rmse`, `ml_roc_auc`, `ml_log_loss`, …), cross-validation splitters as fold-assignment table functions, cross-validated predictions and scores.
- **Hyperparameter tuning:** `ml_tune_agg` with grid, random and successive-halving search.
- **Conformal prediction:** `ml_conformalize_agg`, `ml_predict_interval`, `ml_predict_set`, `ml_conformal_margin`, `ml_coverage`, `ml_interval_score`.
- **Text features:** `ml_hashing_vectorize`.
- **Statistics helpers:** `ml_mutual_info_agg`, `ml_covariance_agg`, `ml_estimate_bandwidth_agg`, `ml_multiple_impute_agg` (aregImpute-style), `ml_spectral_embedding_agg`.
- **Runtime:** fits share one thread pool sized to DuckDB's `threads` (`anofox_ml_threads`). Seeded models are identical whatever the thread count. Per-group memory is capped with `anofox_ml_max_fit_values`.

### Known gaps
See the tracker #8. Some items wait on anofox-ml itself (sipemu/anofox-ml#125): native `predict_log_proba` (currently computed as `ln(predict_proba)`), `kneighbors`, `partial_fit`, stacking and voting, and TPE tuning.

[0.1.0]: https://github.com/DataZooDE/anofox-ml/releases/tag/v0.1.0
