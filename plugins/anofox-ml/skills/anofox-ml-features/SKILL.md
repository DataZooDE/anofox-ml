---
name: anofox-ml-features
description: >
  Preprocessing and unsupervised learning with the anofox_ml DuckDB
  extension: scalers and transformers through ml_fit_agg + ml_transform,
  VARCHAR categories (OneHotEncoder, OrdinalEncoder, TargetEncoder), text
  (ml_hashing_vectorize, CountVectorizer, TfidfVectorizer), imputation
  (SimpleImputer, KnnImputer, IterativeImputer, ml_multiple_impute_agg),
  clustering (ml_cluster_agg, ml_cluster_by), outlier scores
  (ml_outlier_score_agg, ml_outlier_score_by), embeddings and decomposition
  (PCA, t-SNE, ml_fit_transform_agg, ml_spectral_embedding_agg),
  covariance / Mahalanobis, mutual information and feature selection
  kinds. Use for "prepare features", "cluster / segment rows", "find
  anomalies", "reduce dimensions" or "encode strings / text" in SQL.
version: 0.3.0
user-invocable: false
---

# anofox_ml — features, clustering, outliers, embeddings

**Extension:** `anofox_ml` v0.3.0 | Fitting and predicting basics are in the modeling skill.

A transformer is a model like any other: fit it with the unsupervised `ml_fit_agg(kind, [x…] [, params])`, keep the BLOB, and apply it with `ml_transform(model, [x…])`, which returns a `DOUBLE[]`. Chain steps by feeding one `ml_transform` into the next fit, and concatenate blocks with `list_concat`.

## Transformers

| Group | Kinds |
|---|---|
| Scaling | `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `PowerTransformer`, `QuantileTransformer` |
| Construction | `PolynomialFeatures`, `SplineTransformer`, `RcsTransformer`, `KBinsDiscretizer`, `Binarizer`, `Nystroem`, `RbfSampler`, random projections |
| Imputation | `SimpleImputer`, `KnnImputer`, `IterativeImputer`, `MissingIndicator` |
| Selection | `VarianceThreshold`, `SelectKBest`, `GenericUnivariateSelect`, `MutualInformationSelector`, `Rfe`, `Rfecv`, `SequentialFeatureSelector` (the last three take a nested `estimator`) |
| Decomposition | `Pca`, `IncrementalPca`, `KernelPca`, `TruncatedSvd`, `FastIca`, `Nmf`, `SparsePca`, `FactorAnalysis`, `LatentDirichletAllocation` |

`ml_inverse_transform` maps back where the kind supports it; `ml_explained_variance_ratio(model)` reads PCA-like variance.

```sql
CREATE TABLE pipe AS
SELECT s, ml_fit_agg('Pca', ml_transform(s, [price, promo, dow]), '{"n_components": 2}') AS pca
FROM train, (SELECT ml_fit_agg('StandardScaler', [price, promo, dow]) AS s FROM train) GROUP BY s;
SELECT ml_explained_variance_ratio(pca) AS evr, ml_transform(pca, ml_transform(s, [1.0, 0.0, 3.0])) AS z FROM pipe;
```

**Missing values.** A NULL anywhere in the feature list skips the row at fit time and predicts NULL. To let an imputer see a gap, pass it as `NaN`:

```sql
WITH m AS (SELECT ml_fit_agg('SimpleImputer', [coalesce(tenure, 'NaN'::DOUBLE), calls], '{"strategy": "Median"}') AS imp FROM customers)
SELECT ml_transform(imp, ['NaN'::DOUBLE, 2.0]) FROM m;
```

`ml_multiple_impute_agg(id, [x…] [, options] ORDER BY id)` is the exception: a NULL element is the missing value, and it returns `n_impute` completed copies (Hmisc `aregImpute`; options `n_impute`, `nk`, `match`, `kinds`, `seed`).

## Categories and text

- `OneHotEncoder` / `OrdinalEncoder` fit on `VARCHAR` columns directly: `ml_fit_agg('OneHotEncoder', [city, channel] [, params])`, then `ml_transform(enc, [city, channel])`. Params: `handle_unknown` (`"error"` / `"ignore"`), `drop` (`"first"`, `"if_binary"`). NULL categories encode as NULL; `coalesce(col, 'missing')` keeps them.
- `TargetEncoder` needs `y`; use `ml_fit_transform_agg` for its cross-fitted training encoding.
- Text, no fit: `ml_hashing_vectorize(text, n_features [, options])` (dense) or `ml_hashing_vectorize_sparse` (`STRUCT(indices, values)`). Options `alternate_sign` (true), `norm` (`"l2"` / `null`).
- Text with a vocabulary: `ml_fit_agg('CountVectorizer' | 'TfidfVectorizer' | 'HashingVectorizer', body [, params])` over one `VARCHAR` per row, then `ml_transform(vec, body)`. Output is dense, one column per word: bound it with `min_df` / `max_df_frac`. The vocabulary is in `ml_model_fitted`.

```sql
WITH enc AS (SELECT ml_fit_agg('OneHotEncoder', [plan, channel], '{"handle_unknown": "ignore"}') AS e FROM customers),
     m AS (SELECT e, ml_fit_agg('LogisticRegressor', churn, list_concat(ml_transform(e, [plan, channel]), [tenure, calls])) AS model
           FROM customers, enc GROUP BY e)
SELECT ml_predict_label(model, list_concat(ml_transform(e, ['monthly', 'web']), [3.0, 5.0])) FROM m;
```

```sql
WITH v AS (SELECT ml_fit_agg('TfidfVectorizer', body, '{"min_df": 2}') AS vec FROM mails),
     m AS (SELECT vec, ml_fit_agg('LinearSvc', label, ml_transform(vec, body)) AS model FROM mails, v GROUP BY vec)
SELECT ml_predict_label(model, ml_transform(vec, 'free money prize')) FROM m;
```

## Per-row unsupervised results

These aggregates return one entry per input row, **in input order**: put the same `ORDER BY` inside every aggregate of the query and `unnest` side by side.

| Function | Per row |
|---|---|
| `ml_cluster_agg(kind, [x…] [, params [, weight]])` | `BIGINT` cluster label (`-1` = noise for `Dbscan`, `Hdbscan`, `Optics`) |
| `ml_outlier_score_agg(kind, [x…] [, params [, weight]])` | `DOUBLE`, **lower = more anomalous** (`IsolationForest`, `LocalOutlierFactor`, `OneClassSvm`, `EllipticEnvelope`) |
| `ml_fit_transform_agg(kind, [y,] [x…] [, params [, weight]])` | `DOUBLE[]` training embedding (`TSne`, `Isomap`, `LocallyLinearEmbedding`, `ClassicalMds`, …) or encoding |

Clustering kinds: `KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, `Dbscan`, `Hdbscan`, `Optics`, `MeanShift` (bandwidth from `ml_estimate_bandwidth_agg`), `Birch`, `AffinityPropagation`, `AgglomerativeClustering`, `SpectralClustering`, `GaussianMixture`, `BayesianGaussianMixture`. Score a clustering with `ml_silhouette(x, label)`, or against truth with `ml_adjusted_rand` / `ml_normalized_mutual_info`.

```sql
SELECT region, unnest(list(id ORDER BY id)) AS id,
       unnest(ml_cluster_agg('KMeans', [x, y], '{"n_clusters": 3, "seed": 0}' ORDER BY id)) AS cluster
FROM points GROUP BY region ORDER BY region, id LIMIT 5;
```

The table macros do the alignment for you and keep every source column (`id_col` unique per group):

```sql
SELECT region, cluster, count(*) FROM ml_cluster_by('points', region, id, [x, y], 'KMeans', params := '{"n_clusters": 3}')
GROUP BY ALL ORDER BY ALL;
SELECT id, outlier_score FROM ml_outlier_score_by('points', region, id, [x, y], 'IsolationForest') ORDER BY outlier_score LIMIT 3;
```

`ml_fit_transform_by(source, group_col, id_col, x_cols, kind [, params := …])` adds `features DOUBLE[]` the same way.

## Covariance, distances, dependence

| Function | Gives |
|---|---|
| `ml_covariance_agg([x…] [, options])` | `STRUCT(covariance, shrinkage, location)`; `method`: `ledoit_wolf` (default), `oas`, `empirical`, `shrunk`, `mcd` (robust) |
| `ml_fit_agg('MinCovDet' \| 'LedoitWolf' \| …, [x…])` + `ml_mahalanobis(model, [x…])` | squared Mahalanobis distance; ≈ χ²(p) for Gaussian rows, so `> 13.8` flags at 0.999 for p = 2 |
| `ml_covariance(model)` | `STRUCT(location, covariance, precision)` of a covariance kind |
| `ml_mutual_info_agg(y, [x…] [, options])` | MI (nats) per feature, as scikit-learn's `mutual_info_*`; `y` may be `VARCHAR` |
| `ml_spectral_embedding_agg(src, dst [, weight [, options]])` | over graph edges: `STRUCT(node, embedding)[]` |
| `KernelDensity`, `FactorAnalysis` + `ml_score_samples` | per-row log-likelihood |

```sql
WITH m AS (SELECT region, ml_fit_agg('MinCovDet', [x, y]) AS model FROM points GROUP BY region)
SELECT region, count(*) FILTER (WHERE ml_mahalanobis(model, [x, y]) > 13.8) AS outliers
FROM points JOIN m USING (region) GROUP BY region ORDER BY region;
```
