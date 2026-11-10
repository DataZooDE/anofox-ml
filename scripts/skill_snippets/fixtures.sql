-- Tables every ```sql block in plugins/anofox-ml/skills runs against
-- (scripts/check_skill_snippets.py prepends this file).
SET anofox_telemetry_enabled = false;

-- store sales: 3 stores x 1000 days; units depend on weekday, price, promo
CREATE TABLE sales AS
SELECT *, 100 + 10 * dow - 20 * price + 30 * promo + 5 * noise AS units,
       (100 + 10 * dow - 20 * price + 30 * promo + 5 * noise) * price AS revenue
FROM (SELECT i AS id, 'store_' || (i % 3) AS store, i // 3 AS day, (i // 3) % 7 AS dow,
             0.5 + (hash(i, 'p') % 1000) / 1000 AS price,
             (hash(i, 'q') % 5 = 0)::INT AS promo,
             (hash(i, 'n') % 1000) / 1000 AS noise
      FROM range(3000) t(i));
CREATE TABLE train AS SELECT * FROM sales WHERE day < 800;
CREATE TABLE test AS SELECT * FROM sales WHERE day >= 800;

-- customers: churn depends on tenure and support calls
CREATE TABLE customers AS
SELECT *, CASE WHEN churned = 1 THEN 'yes' ELSE 'no' END AS churn,
       CASE WHEN tenure < 12 THEN 'monthly' WHEN tenure < 36 THEN 'annual' ELSE 'two_year' END AS plan,
       CASE id % 3 WHEN 0 THEN 'web' WHEN 1 THEN 'store' ELSE 'phone' END AS channel
FROM (SELECT i AS id, (hash(i, 't') % 60)::DOUBLE AS tenure, (hash(i, 'c') % 8)::DOUBLE AS calls,
             (hash(i, 's') % 100)::DOUBLE / 10 AS spend,
             ((hash(i, 't') % 60) < 15 AND (hash(i, 'c') % 8) > 2 OR hash(i, 'r') % 10 = 0)::INT::DOUBLE AS churned
      FROM range(1000) t(i));

-- 2-D points in three blobs, per region
CREATE TABLE points AS
SELECT i AS id, 'r' || (i % 2) AS region,
       (i % 3) * 5 + (hash(i, 'x') % 100) / 50.0 AS x,
       (i % 3) * 3 + (hash(i, 'y') % 100) / 50.0 AS y
FROM range(600) t(i);

-- short labelled documents
CREATE TABLE mails AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'spam' ELSE 'ham' END AS label,
       CASE WHEN i % 2 = 0 THEN 'win free money now ' || (i % 7) || ' prize offer'
            ELSE 'meeting notes for project ' || (i % 5) || ' see agenda' END AS body
FROM range(200) t(i);
