-- Tables every ```sql block of README.md and plugins/anofox-ml/skills runs
-- against (scripts/check_skill_snippets.py prepends this file; each block
-- gets a fresh database).
SET anofox_telemetry_enabled = false;
-- json (json_extract) is built into stock CLIs; dev builds load an installed copy.
SET autoinstall_known_extensions = false;
SET autoload_known_extensions = true;

-- store sales: 3 stores x 1000 days; units depend on weekday, price, promo.
-- x1..x5 / a / b / y / sales are the README's generic names for the same data.
CREATE TABLE sales AS
SELECT *, units AS sales, units AS y, price AS x1, promo::DOUBLE AS x2, dow::DOUBLE AS x3, noise AS x4,
       (hash(id, 'x5') % 100) / 100.0 AS x5, price AS a, dow::DOUBLE AS b,
       CASE WHEN units > 130 THEN 'high' ELSE 'low' END AS label,
       ['berlin', 'paris', 'rome'][1 + id % 3] AS city, ['web', 'shop'][1 + id % 2] AS channel,
       (id % 3) AS store_code, 1.0 + (id % 4) / 4.0 AS weight
FROM (SELECT *, 100 + 10 * dow - 20 * price + 30 * promo + 5 * noise AS units,
             (100 + 10 * dow - 20 * price + 30 * promo + 5 * noise) * price AS revenue
      FROM (SELECT i AS id, 'store_' || (i % 3) AS store, i // 3 AS day, (i // 3) % 7 AS dow,
                   0.5 + (hash(i, 'p') % 1000) / 1000 AS price,
                   (hash(i, 'q') % 5 = 0)::INT AS promo,
                   (hash(i, 'n') % 1000) / 1000 AS noise
            FROM range(3000) t(i)));
CREATE TABLE train AS SELECT * FROM sales WHERE day < 800;
CREATE TABLE test AS SELECT * FROM sales WHERE day >= 800;
CREATE TABLE calibration AS SELECT * FROM sales WHERE day BETWEEN 600 AND 799;
CREATE TABLE new_rows AS SELECT * FROM sales WHERE day >= 950;
CREATE TABLE plan AS SELECT DISTINCT price, promo FROM test LIMIT 20;
CREATE TABLE grid AS SELECT unnest(range(0, 7))::DOUBLE AS x;
CREATE TABLE models AS
SELECT store, ml_fit_agg('HistGradientBoostingRegressor', units, [price, promo, dow], '{"n_estimators": 50}') AS model
FROM train GROUP BY store;

-- customers: churn depends on tenure and support calls
CREATE TABLE customers AS
SELECT *, CASE WHEN churned = 1 THEN 'yes' ELSE 'no' END AS churn,
       CASE WHEN tenure < 12 THEN 'monthly' WHEN tenure < 36 THEN 'annual' ELSE 'two_year' END AS plan,
       CASE id % 3 WHEN 0 THEN 'web' WHEN 1 THEN 'store' ELSE 'phone' END AS channel,
       tenure AS a, calls AS b, spend AS c, tenure AS x1, calls AS x2
FROM (SELECT i AS id, (hash(i, 't') % 60)::DOUBLE AS tenure, (hash(i, 'c') % 8)::DOUBLE AS calls,
             (hash(i, 's') % 100)::DOUBLE / 10 AS spend,
             ((hash(i, 't') % 60) < 15 AND (hash(i, 'c') % 8) > 2 OR hash(i, 'r') % 10 = 0)::INT::DOUBLE AS churned
      FROM range(1000) t(i));
CREATE TABLE churn_model AS SELECT ml_fit_agg('LogisticRegressor', churned, [x1, x2]) AS model FROM customers;

-- iris-like: three species separated by two measurements
CREATE TABLE iris AS
SELECT ['setosa', 'versicolor', 'virginica'][1 + i % 3] AS species,
       4.5 + (i % 3) * 1.2 + (hash(i, 'l') % 100) / 200.0 AS sepal_length,
       0.2 + (i % 3) * 0.9 + (hash(i, 'w') % 100) / 300.0 AS petal_width
FROM range(150) t(i);

-- 2-D points in three blobs, per region (x1..x3 for the embedding examples)
CREATE TABLE points AS
SELECT *, x AS x1, y AS x2, x + y AS x3
FROM (SELECT i AS id, 'r' || (i % 2) AS region,
             (i % 3) * 5 + (hash(i, 'x') % 100) / 50.0 AS x,
             (i % 3) * 3 + (hash(i, 'y') % 100) / 50.0 AS y
      FROM range(600) t(i));

-- short labelled documents
CREATE TABLE mails AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'spam' ELSE 'ham' END AS label,
       CASE WHEN i % 2 = 0 THEN 'win free money now ' || (i % 7) || ' prize offer'
            ELSE 'meeting notes for project ' || (i % 5) || ' see agenda' END AS body
FROM range(200) t(i);
CREATE TABLE new_mails AS SELECT * FROM mails WHERE id < 10;

-- positive skewed target
CREATE TABLE shops AS SELECT exp(1 + 0.3 * x1 + 0.1 * x2) AS revenue, x1, x2
FROM (SELECT (i % 10)::DOUBLE AS x1, (hash(i) % 7)::DOUBLE AS x2 FROM range(300) t(i));

-- insurance claims with exposure
CREATE TABLE policies AS
SELECT (hash(i, 'k') % 3)::DOUBLE AS claims, (20 + i % 50)::DOUBLE AS age, (i % 4)::DOUBLE AS region_code,
       1 + (i % 4) AS policy_years
FROM range(500) t(i);
CREATE TABLE new_policies AS SELECT * FROM policies LIMIT 10;

-- two blocks for CCA
CREATE TABLE t AS
SELECT x1, x2, x3, x1 + 0.5 * x2 + (hash(i, 'u') % 10) / 10.0 AS y1, x3 - x1 + (hash(i, 'v') % 10) / 10.0 AS y2
FROM (SELECT i, (i % 11)::DOUBLE AS x1, (hash(i, 'a') % 13)::DOUBLE AS x2, (hash(i, 'b') % 7)::DOUBLE AS x3 FROM range(300) t(i));

-- sensor readings
CREATE TABLE readings AS
SELECT i AS id, 's' || (i % 3) AS sensor, 'm' || (i % 2) AS machine,
       20 + (hash(i, 'T') % 100) / 20.0 AS temp, 1 + (hash(i, 'P') % 100) / 100.0 AS pressure,
       (hash(i, 'V') % 100) / 50.0 AS vibration
FROM range(600) t(i);

-- events per ISO week; latency shifts in the last week
CREATE TABLE events AS
SELECT i AS id, '2026-W' || (40 + i // 100) AS week,
       10 + (hash(i, 'l') % 50) / 10.0 + CASE WHEN i >= 500 THEN 3 ELSE 0 END AS latency,
       (hash(i, 's') % 1000)::DOUBLE AS size, (hash(i, 'r') % 3)::DOUBLE AS retries
FROM range(600) t(i);

-- a deployed model's labelled reference rows and unlabelled production rows
CREATE TABLE reference AS SELECT *, units + (hash(id, 'e') % 100) / 50.0 - 1 AS forecast, 'w' || (day // 100) AS week FROM train;
CREATE TABLE production AS SELECT *, units + (hash(id, 'e') % 100) / 50.0 - 1 AS forecast, 'w' || (day // 50) AS week FROM test;

-- search results: relevance 0..3 per (query, document)
CREATE TABLE search_log AS
SELECT i // 10 AS query_id, x1, x2, least(3, greatest(0, floor(x1 * 4 - x2 * 2 + 1)))::DOUBLE AS relevance
FROM (SELECT i, (hash(i, 'a') % 1000) / 1000.0 AS x1, (hash(i, 'b') % 1000) / 1000.0 AS x2 FROM range(1000) t(i));
