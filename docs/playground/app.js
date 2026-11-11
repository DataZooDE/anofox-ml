// anofox_ml playground: DuckDB-Wasm + the anofox_ml extension, served next to
// this page under ext/<duckdb version>/<platform>/ (see .github/workflows/pages.yml).
//
// Query parameters (used by the browser test, test/wasm/browser.mjs):
//   ?dist=<url>  where the @duckdb/duckdb-wasm wasm and worker files are
//                (default: jsDelivr; the JS module always comes from jsDelivr's
//                +esm build, which resolves its apache-arrow import)
//   ?ext=<url>   extension repository (default: ext/ next to this page)

const DUCKDB_WASM = '1.33.1-dev114.0'; // DuckDB v2.0.0
const params = new URLSearchParams(location.search);
const DIST = params.get('dist') || `https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@${DUCKDB_WASM}/dist/`;
const EXT_REPO = params.get('ext') || new URL('ext', location.href).href;

const EXAMPLES = {
  'Fit one model per store': `-- 3 stores; units depend on weekday, price and promotion
CREATE OR REPLACE TABLE sales AS
SELECT *, 100 + 10 * dow - 20 * price + 30 * promo + 5 * noise AS units
FROM (SELECT i AS id, 'store_' || (i % 3) AS store, i % 7 AS dow,
             0.5 + (hash(i, 'p') % 1000) / 1000 AS price,
             (hash(i, 'q') % 5 = 0)::INT AS promo, (hash(i, 'n') % 1000) / 1000 AS noise
      FROM range(1500) t(i));

-- one gradient-boosting model per store, stored as a BLOB
CREATE OR REPLACE TABLE models AS
SELECT store, ml_fit_agg('HistGradientBoostingRegressor', units, [price, promo, dow],
                         '{"n_estimators": 100}') AS model
FROM sales WHERE id < 1200 GROUP BY store;

-- evaluate on the held-out rows
SELECT store, round(ml_rmse(units, ml_predict(model, [price, promo, dow])), 3) AS rmse,
       round(ml_r2(units, ml_predict(model, [price, promo, dow])), 3) AS r2
FROM sales JOIN models USING (store) WHERE id >= 1200
GROUP BY store ORDER BY store;`,

  'Classify with string labels': `CREATE OR REPLACE TABLE customers AS
SELECT i AS id, (hash(i, 't') % 60)::DOUBLE AS tenure, (hash(i, 'c') % 8)::DOUBLE AS calls,
       CASE WHEN (hash(i, 't') % 60) < 15 AND (hash(i, 'c') % 8) > 2 THEN 'churn' ELSE 'stay' END AS label
FROM range(800) t(i);

WITH m AS (SELECT ml_fit_agg('RandomForestClassifier', label, [tenure, calls], '{"n_estimators": 30, "seed": 0}') AS model
           FROM customers WHERE id < 600)
SELECT ml_model_info(model).labels AS labels,
       round(ml_accuracy(label, ml_predict_label(model, [tenure, calls])), 3) AS accuracy,
       ml_predict_proba(model, [5.0, 6.0]) AS p_new_customer
FROM customers, m WHERE id >= 600 GROUP BY model;`,

  'Clusters and outliers': `CREATE OR REPLACE TABLE points AS
SELECT i AS id, (i % 3) * 5 + (hash(i, 'x') % 100) / 50.0 AS x, (i % 3) * 3 + (hash(i, 'y') % 100) / 50.0 AS y
FROM range(300) t(i);

-- one label and one outlier score per row; ORDER BY keeps them aligned with id
SELECT unnest(list(id ORDER BY id)) AS id,
       unnest(ml_cluster_agg('KMeans', [x, y], '{"n_clusters": 3, "seed": 0}' ORDER BY id)) AS cluster,
       round(unnest(ml_outlier_score_agg('IsolationForest', [x, y], '{"seed": 0}' ORDER BY id)), 3) AS outlier_score
FROM points ORDER BY outlier_score LIMIT 10;`,

  'Tune hyperparameters (TPE)': `CREATE OR REPLACE TABLE s AS
SELECT i, i / 300 * 6.0 AS x, sin(i / 300 * 6.0) + ((i * 7919) % 101) / 101.0 - 0.5 AS y FROM range(300) t(i);

-- Bayesian search over n_neighbors with 4-fold cross-validation
SELECT t.best_params, round(t.best_score, 4) AS cv_rmse
FROM (SELECT ml_tune_agg('KnnRegressor', y, [x], '{"n_neighbors": {"int": [1, 150], "log": true}}',
        '{"search": "tpe", "n_trials": 20, "metric": "rmse", "cv": 4, "seed": 1}' ORDER BY i) AS t FROM s);`,

  'Prediction intervals': `CREATE OR REPLACE TABLE d AS
SELECT i AS id, (i % 50)::DOUBLE AS x, 2 * (i % 50) + (hash(i) % 100) / 10.0 AS y FROM range(1500) t(i);

-- fit on the first rows, calibrate on the next, check coverage on the rest
CREATE OR REPLACE TABLE cal AS
SELECT ml_conformalize_agg(m, y, [x]) AS m
FROM d, (SELECT ml_fit_agg('RidgeRegressor', y, [x]) AS m FROM d WHERE id < 800) WHERE id BETWEEN 800 AND 1199 GROUP BY ALL;

SELECT round(ml_coverage(y, pi.lower, pi.upper), 3) AS coverage_90, round(avg(pi.upper - pi.lower), 2) AS mean_width
FROM (SELECT y, ml_predict_interval(m, [x], 0.1) AS pi FROM d, cal WHERE id >= 1200);`,

  'What estimators are there?': `SELECT kind, capabilities FROM ml_estimators() ORDER BY kind;`,
};

const $ = (id) => document.getElementById(id);
let conn = null;

export async function boot({ dist = DIST, extRepo = EXT_REPO } = {}) {
  const duckdb = await import(`https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@${DUCKDB_WASM}/+esm`);
  const bundles = {
    base: { mainModule: `${dist}duckdb-base.wasm`, mainWorker: `${dist}duckdb-browser-base.worker.js` },
    threads: {
      mainModule: `${dist}duckdb-threads.wasm`,
      mainWorker: `${dist}duckdb-browser-threads.worker.js`,
      pthreadWorker: `${dist}duckdb-browser-threads.pthread.worker.js`,
    },
  };
  const bundle = await duckdb.selectBundle(bundles);
  // A worker from another origin has to be started from a same-origin blob.
  const workerUrl = new URL(bundle.mainWorker, location.href).origin === location.origin
    ? bundle.mainWorker
    : URL.createObjectURL(new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' }));
  const worker = new Worker(workerUrl);
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  await db.open({ allowUnsignedExtensions: true });
  const c = await db.connect();
  await c.query(`INSTALL anofox_ml FROM '${extRepo}'`);
  await c.query('LOAD anofox_ml');
  const v = (await c.query(
    "SELECT version() AS duckdb, ml_version() AS anofox_ml, (SELECT platform FROM pragma_platform()) AS platform",
  )).toArray()[0].toJSON();
  return { db, conn: c, versions: v };
}

function render(table) {
  const out = $('result');
  out.hidden = false;
  const fields = table.schema.fields.map((f) => f.name);
  const rows = table.toArray().slice(0, 500);
  const cell = (v) => {
    if (v === null || v === undefined) return ['NULL', false];
    if (typeof v === 'number' || typeof v === 'bigint') return [String(v), true];
    if (v && typeof v.toJSON === 'function') return [JSON.stringify(v.toJSON()), false];
    if (v instanceof Uint8Array) return [`<BLOB ${v.length} bytes>`, false];
    return [typeof v === 'object' ? JSON.stringify(v, (_, x) => (typeof x === 'bigint' ? Number(x) : x)) : String(v), false];
  };
  const esc = (s) => s.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
  out.innerHTML = `<table><thead><tr>${fields.map((f) => `<th>${esc(f)}</th>`).join('')}</tr></thead><tbody>${
    rows.map((r) => `<tr>${fields.map((f) => { const [t, n] = cell(r[f]); return `<td${n ? ' class="num"' : ''}>${esc(t)}</td>`; }).join('')}</tr>`).join('')
  }</tbody></table>`;
  return `${table.numRows} row${table.numRows === 1 ? '' : 's'}${table.numRows > 500 ? ' (first 500 shown)' : ''}`;
}

async function run() {
  if (!conn) return;
  const status = $('status');
  status.className = '';
  status.textContent = 'Running…';
  $('run').disabled = true;
  const t0 = performance.now();
  try {
    const table = await conn.query($('sql').value);
    const rows = render(table);
    status.textContent = `${rows} in ${((performance.now() - t0) / 1000).toFixed(2)} s`;
  } catch (e) {
    status.className = 'error';
    status.textContent = String(e && e.message ? e.message : e);
    $('result').hidden = true;
  } finally {
    $('run').disabled = false;
  }
}

function setup() {
  const sel = $('example');
  for (const name of Object.keys(EXAMPLES)) sel.add(new Option(name, name));
  const pick = () => { $('sql').value = EXAMPLES[sel.value]; };
  sel.addEventListener('change', pick);
  pick();
  $('run').addEventListener('click', run);
  $('sql').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); }
  });
}

if (!params.has('test')) {
  setup();
  boot().then(({ conn: c, versions }) => {
    conn = c;
    $('run').disabled = false;
    $('status').textContent = 'Ready. Pick an example or write SQL, then Run.';
    $('versions').textContent = `DuckDB ${versions.duckdb} (${versions.platform}), anofox_ml ${versions.anofox_ml}.`;
  }).catch((e) => {
    $('status').className = 'error';
    $('status').textContent = `Could not start: ${e && e.message ? e.message : e}`;
  });
}
