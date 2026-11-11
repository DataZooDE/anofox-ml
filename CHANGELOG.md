# Changelog

## [0.2.0] - 2026-11-11

Built for **DuckDB v1.5.6** and **v2.0.0**, with [anofox-ml](https://github.com/sipemu/anofox-ml) **0.5** from crates.io. This is the version submitted as a DuckDB community extension (#4). The release binaries are unsigned; see [Installation](https://github.com/DataZooDE/anofox-ml#installation).

### Fixed
- DuckDB-Wasm: fits and predictions work. The wasm builds loaded, but every fit and prediction failed, because DuckDB-Wasm cannot start the threads of the extension's thread pools; they now run on the current thread there (#84).

### Changed
- Errors from a panic caught in the Rust core include the panic message.
- `PerpetualRegressor` / `PerpetualClassifier` are not in the wasm builds: the perpetual crate starts its own threads on every fit (#84).
- `ml_inverse_transform` no longer rejects inputs of the transformed width (PCA components, one-hot columns); the model checks the width itself (#11).
- anofox-ml 0.5: 188 estimator kinds (from 144), among them voting / stacking / calibration / transformed-target / multi-output meta-estimators, Gaussian processes, PLS / CCA, multi-task lasso, LARS CV, text vectorizers and more. Stored models from 0.2 and 0.4 load and predict identically. `ml_predict_log_proba` is native where the estimator has it (finite where probabilities underflow).
- Building from source needs Rust 1.88 (was 1.87), for the PerpetualBooster kinds (#47).

### Added
- Claude Code plugin with four skills (`anofox-ml-modeling`, `-features`, `-evaluation`, `-advanced`): `/plugin marketplace add DataZooDE/anofox-ml`. `make check_skills` runs every SQL example in them against the built extension.
- DuckDB-Wasm: CI runs the SQL tests in DuckDB-Wasm (`test/wasm`; `wasm_eh` on DuckDB 1.5.6 and 2.0), and the wasm builds cover DuckDB 2.0 as well. WASM builds pin their Rust toolchain in CMake, so the community-extensions CI builds them too (#84).
- `ml_tune_agg` with `search: "tpe"`: Bayesian optimisation with a tree-structured Parzen estimator (Optuna's univariate TPE), seeded and thread-count independent (#5).
- DuckDB-Wasm: the wasm builds link the Rust core (the 0.1.0 wasm builds didn't load) and releases attach them again; CI checks the module has no unresolved FFI imports (#29).
- Text vectorizers on `VARCHAR` documents: `ml_fit_agg(kind, text [, params])` for `CountVectorizer`, `TfidfVectorizer` and `HashingVectorizer`, and `ml_transform(model, text)`, matching scikit-learn (#14).
- Meta-estimators through `ml_fit_agg`, with tests against scikit-learn and recipes: `VotingRegressor` / `VotingClassifier` (weights, hard / soft voting), `StackingRegressor` / `StackingClassifier`, `CalibratedClassifierCV` and `TransformedTargetRegressor` (#12).
- Multi-target kinds (`MultiOutputRegressor`, `RegressorChain`, `MultiTaskLasso`, …) fitted on `[X | Y]` with `y_columns`, and `ml_predict_targets(model, x)` for their per-target predictions (#11).
- `ml_predict_std(model, x)`: predictive standard deviation of `GaussianProcessRegressor` (#11).
- `OlsRegressor` / `RidgeRegressor` with `inference: true`: prediction intervals through `ml_predict_interval` and the coefficient table in `ml_model_fitted` (#11).
- Tests for the new kinds `LassoLarsIC`, `BinomialRegressor`, `IsotonicRegressor`, `GaussianProcessClassifier`, numeric `OneHotEncoder` and `BaggingEstimator*` (#11).
- GLM offsets / exposure: `ml_fit_with_offset_agg(kind, y, x, offset [, params [, weight]])` and `ml_predict(model, x, offset)` for Poisson, Gamma, Tweedie and Binomial regression (#18).
- `ml_transform_y(model, y)`: Y scores of `Cca`, `PlsCanonical` and `PlsSvd`. Models fitted on `[X | Y]` (`y_columns`) store the width of X, so `ml_transform` / `ml_predict` take X alone (#18).
- The feature selection kinds `Rfe`, `Rfecv` and `SequentialFeatureSelector` (nested estimator) work through `ml_fit_agg` / `ml_transform`, matching scikit-learn (#18).
- `ml_partial_fit_agg`: incremental batches for naive Bayes, SGD, PassiveAggressive, MLP, the scalers and IncrementalPca; MLP warm start through `ml_warm_start_agg` (#17).
- `ml_kneighbors(model, x [, k])`: nearest training rows of the kNN models (#16).
- DuckDB 2.0 support: the same source builds and passes the tests against DuckDB v1.5.6 and v2.0.0; CI builds both and releases attach binaries for both (#66).
- `ml_feature_importances(model)` for tree and boosting kinds; `n_iter`, `inertia` and `oob_score` in `ml_model_info` (#10).
- `ml_predict_interval` / `ml_predict_set` for the conformal kinds: split, CV+ / jackknife+, normalised, CQR, Mondrian and classifier sets (#6). New aggregates `ml_set_coverage`, `ml_mean_set_size` and `ml_mean_pinball_loss`.
- Tests that model BLOBs saved with anofox-ml 0.2 still load and predict the same (#48).
- `ml_staged_predict(model, x)` for the boosting kinds and `ml_fit_with_validation_agg` for early stopping on an explicit validation set (#46).
- `ml_warm_start_agg(model, y, x [, params])`: continue training a stored gradient-boosting / HGB / LGBM model (#17).
- Covariance: `ml_covariance_agg` methods `shrunk` and `mcd` (robust) and a `location` field; `ml_covariance(model)` and `ml_mahalanobis(model, x)` for fitted covariance estimators (#45).
- TF-IDF recipe: `TfidfTransformer` on hashed counts, tested against scikit-learn (#14).
- `ml_estimators()` has a `capabilities` column; calling a method a model lacks names what it supports and which kinds have the method (#9).
- `ml_rfe_agg`: recursive feature elimination, matching scikit-learn's `RFE` (#18).
- `ml_domain_classifier_drift_by`: multivariate drift per chunk by a domain classifier's cross-validated ROC AUC (#43).
- `ml_dle_estimate_by`: direct loss estimation of MAE / RMSE per chunk without labels (#42).
- PerpetualBooster: `PerpetualRegressor` / `PerpetualClassifier` (one `budget` parameter instead of learning rate and tree count), also as the base model of CQR (#47).
- Semi-supervised fits: for `LabelPropagation`, `LabelSpreading` and `SelfTrainingClassifier`, rows with a NULL target are unlabelled rows of the fit instead of being skipped (#44).

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
