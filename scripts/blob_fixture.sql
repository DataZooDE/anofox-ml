-- Write the model BLOB fixture of the current build (test/data/blobs):
--   build/release/duckdb < scripts/blob_fixture.sql
-- writes models_new.csv / expected_new.csv in the working directory; rename
-- them to models_<version>.csv / expected_<version>.csv, where <version> is
-- the anofox-ml version being replaced, before bumping it.
--
-- Training data as in test/data/blobs/README.md: 60 rows,
-- x = [i / 10, (7i mod 11) / 3], y = 2·x1 − x2 + (13i mod 5) / 10.
SET anofox_telemetry_enabled = false;

CREATE TABLE d AS
SELECT i, [i / 10, (7 * i % 11) / 3] AS x, 2 * (i / 10) - (7 * i % 11) / 3 + (13 * i % 5) / 10 AS y,
       ['hi', 'lo', 'mid'][1 + i % 3] AS lab,
       ['berlin', 'paris', 'rome'][1 + i % 3] AS city, ['web', 'shop'][1 + i % 2] AS chan,
       ['red apples and pears', 'green pears', 'red wine and cheese', 'cheese and apples'][1 + i % 4] AS doc
FROM range(60) t(i);

-- (case, kind, method, model); method says which function the test calls
CREATE TABLE m AS
          SELECT 'ols' AS c, 'OlsRegressor' AS k, 'predict' AS meth, ml_fit_agg('OlsRegressor', y, x) AS model FROM d
UNION ALL SELECT 'logistic_labels', 'LogisticRegressor', 'predict_proba', ml_fit_agg('LogisticRegressor', lab, x) FROM d
UNION ALL SELECT 'rf', 'RandomForestRegressor', 'predict', ml_fit_agg('RandomForestRegressor', y, x, '{"n_estimators": 10, "seed": 0}') FROM d
UNION ALL SELECT 'hgb', 'HistGradientBoostingRegressor', 'predict', ml_fit_agg('HistGradientBoostingRegressor', y, x, '{"n_estimators": 20}') FROM d
UNION ALL SELECT 'gnb', 'GaussianNB', 'predict_proba', ml_fit_agg('GaussianNB', (y > 2)::INT::DOUBLE, x) FROM d
UNION ALL SELECT 'kmeans', 'KMeans', 'predict', ml_fit_agg('KMeans', x, '{"n_clusters": 3, "seed": 0}') FROM d
UNION ALL SELECT 'scaler', 'StandardScaler', 'transform', ml_fit_agg('StandardScaler', x) FROM d
UNION ALL SELECT 'pca', 'Pca', 'transform', ml_fit_agg('Pca', x) FROM d
UNION ALL SELECT 'iforest', 'IsolationForest', 'score_samples', ml_fit_agg('IsolationForest', x, '{"seed": 0}') FROM d
UNION ALL SELECT 'catboost', 'CatBoostRegressor', 'predict', ml_fit_agg('CatBoostRegressor', y, x, '{"iterations": 20}') FROM d
UNION ALL SELECT 'catboost_clf', 'CatBoostClassifier', 'predict_proba', ml_fit_agg('CatBoostClassifier', lab, x, '{"iterations": 20}') FROM d
UNION ALL SELECT 'lgbm', 'LgbmRegressor', 'predict', ml_fit_agg('LgbmRegressor', y, x, '{"n_estimators": 20}') FROM d
UNION ALL SELECT 'lgbm_clf', 'LgbmClassifier', 'predict_proba', ml_fit_agg('LgbmClassifier', lab, x, '{"n_estimators": 20}') FROM d
UNION ALL SELECT 'split_conformal', 'SplitConformalRegressor', 'predict', ml_fit_agg('SplitConformalRegressor', y, x) FROM d
UNION ALL SELECT 'ledoit_wolf', 'LedoitWolf', 'transform', ml_fit_agg('LedoitWolf', x) FROM d
UNION ALL SELECT 'perpetual', 'PerpetualRegressor', 'predict', ml_fit_agg('PerpetualRegressor', y, x, '{"budget": 0.5}') FROM d
UNION ALL SELECT 'voting', 'VotingRegressor', 'predict', ml_fit_agg('VotingRegressor', y, x, '{"estimators": [{"estimator": "OlsRegressor", "estimator_params": {}}, {"estimator": "RandomForestRegressor", "estimator_params": {"n_estimators": 5, "seed": 0}}]}') FROM d
UNION ALL SELECT 'gp', 'GaussianProcessRegressor', 'predict', ml_fit_agg('GaussianProcessRegressor', y, x) FROM d
-- VARCHAR inputs: x is a VARCHAR[] (encoders) or one document (text kinds)
UNION ALL SELECT 'onehot', 'OneHotEncoder', 'transform_varchar', ml_fit_agg('OneHotEncoder', [city, chan]) FROM d
UNION ALL SELECT 'ordinal', 'OrdinalEncoder', 'transform_varchar', ml_fit_agg('OrdinalEncoder', [city, chan]) FROM d
UNION ALL SELECT 'tfidf', 'TfidfVectorizer', 'transform_text', ml_fit_agg('TfidfVectorizer', doc) FROM d;

COPY (SELECT c AS "case", k AS kind, meth AS method, hex(model) AS blob_hex FROM m ORDER BY c)
TO 'models_new.csv' (HEADER, QUOTE '"');

CREATE TABLE rows_num AS SELECT * FROM (VALUES (0, [0.5, 1.0]), (1, [2.0, 3.0]), (2, [4.5, 0.2]), (3, [5.9, 3.3])) v(r, x);
CREATE TABLE rows_str AS SELECT * FROM (VALUES (0, ['berlin', 'web']), (1, ['paris', 'shop']), (2, ['rome', 'web']), (3, ['berlin', 'shop'])) v(r, x);
CREATE TABLE rows_doc AS SELECT * FROM (VALUES (0, 'red apples'), (1, 'green cheese and wine'), (2, 'pears'), (3, 'unknown words')) v(r, x);

COPY (
    SELECT c AS "case", r AS "row", x::VARCHAR AS x, (CASE meth
        WHEN 'predict' THEN [ml_predict(model, x)]
        WHEN 'predict_proba' THEN ml_predict_proba(model, x)
        WHEN 'transform' THEN ml_transform(model, x)
        WHEN 'score_samples' THEN ml_score_samples(model, x) END)::VARCHAR AS expected
    FROM m, rows_num WHERE meth IN ('predict', 'predict_proba', 'transform', 'score_samples')
    UNION ALL
    SELECT c, r, x::VARCHAR, ml_transform(model, x)::VARCHAR FROM m, rows_str WHERE meth = 'transform_varchar'
    UNION ALL
    SELECT c, r, x, ml_transform(model, x)::VARCHAR FROM m, rows_doc WHERE meth = 'transform_text'
    ORDER BY 1, 2
) TO 'expected_new.csv' (HEADER, QUOTE '"');
