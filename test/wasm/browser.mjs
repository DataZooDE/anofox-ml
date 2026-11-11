#!/usr/bin/env node
// Browser runtime test of the wasm builds (#89): headless Chromium runs
// test/sql inside the playground page (docs/playground), with DuckDB-Wasm 2.0.
//
// One local origin serves the page, the @duckdb/duckdb-wasm dist files, the
// extension (as ext/v2.0.0/<platform>/) and test/sql. With --threads it sends
// COOP / COEP, so the page is cross-origin isolated and DuckDB-Wasm picks its
// threads bundle (wasm_threads); without, the single-threaded one (wasm_eh).
//
// --threads is for when threaded DuckDB-Wasm can load extensions: today no
// wasm_threads extension loads, DuckDB's own included ("mismatch in shared
// state of memory"; duckdb/extension-ci-tools#453, duckdb/duckdb-wasm#2041).
//
// Usage: node browser.mjs --ext <anofox_ml.duckdb_extension.wasm> [--threads] [file.test ...]
// Needs @duckdb/duckdb-wasm 1.33.1-dev114.0 (DuckDB 2.0) in node_modules and
// `npx playwright install chromium`.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const require = createRequire(path.join(HERE, 'package.json'));
const DIST = path.dirname(require.resolve('@duckdb/duckdb-wasm/dist/duckdb-node.cjs'));
// As in run.mjs: no host filesystem, and no PerpetualBooster in the wasm builds.
const SKIP = new Set(['test/sql/ml_blob_compat.test', 'test/sql/ml_perpetual.test']);

const args = { ext: null, threads: false, files: [] };
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a === '--ext') args.ext = path.resolve(process.argv[++i]);
  else if (a === '--threads') args.threads = true;
  else if (a.endsWith('.test')) args.files.push(a);
}
if (!args.ext) throw new Error('pass --ext <path to anofox_ml.duckdb_extension.wasm>');
const platform = args.threads ? 'wasm_threads' : 'wasm_eh';
const files = (args.files.length ? args.files
  : fs.readdirSync(path.join(REPO, 'test/sql')).filter((f) => f.endsWith('.test')).sort().map((f) => `test/sql/${f}`))
  .filter((f) => !SKIP.has(f));

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.test': 'text/plain' };
function resolve(url) {
  if (url === '/' || url === '/index.html') return path.join(REPO, 'docs/playground/index.html');
  if (url === '/app.js') return path.join(REPO, 'docs/playground/app.js');
  if (url === '/sqllogic.mjs') return path.join(HERE, 'sqllogic.mjs');
  if (url.startsWith('/dist/')) return path.join(DIST, url.slice(6));
  if (url.startsWith('/test/sql/')) return path.join(REPO, url.slice(1));
  if (url.startsWith('/ext/') && url.endsWith('/anofox_ml.duckdb_extension.wasm')) return args.ext;
  return null;
}
const server = http.createServer((req, res) => {
  const file = resolve(decodeURIComponent((req.url || '/').split('?')[0]));
  if (!file || !fs.existsSync(file)) { res.statusCode = 404; return res.end('not found'); }
  res.setHeader('Content-Type', TYPES[path.extname(file)] || 'application/octet-stream');
  if (args.threads) {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  }
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch();
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') process.stdout.write(`  console: ${m.text().slice(0, 300)}\n`); });
  page.on('pageerror', (e) => process.stdout.write(`  pageerror: ${e.message}\n`));
  await page.exposeFunction('report', (s) => process.stdout.write(s + '\n'));
  await page.goto(`${origin}/?test&dist=${encodeURIComponent(`${origin}/dist/`)}&ext=${encodeURIComponent(`${origin}/ext`)}`);
  const totals = await page.evaluate(async ({ files, platform }) => {
    const { boot } = await import('/app.js');
    const { parseTest, runRecords } = await import('/sqllogic.mjs');
    const { db, conn, versions } = await boot();
    await window.report(`DuckDB ${versions.duckdb} (${versions.platform}, isolated: ${self.crossOriginIsolated}), anofox_ml ${versions.anofox_ml}`);
    if (versions.platform !== platform) throw new Error(`expected platform ${platform}, got ${versions.platform}`);
    await conn.close();
    const t = { passed: 0, failed: 0, skipped: 0, failedFiles: [] };
    for (const file of files) {
      // A fresh catalog per file, as sqllogictest does.
      await db.open({ allowUnsignedExtensions: true });
      const c = await db.connect();
      await c.query(`INSTALL anofox_ml FROM '${new URL('/ext', location.href).href}'`);
      await c.query('LOAD anofox_ml');
      const rows = async (sql) => (await c.query(sql)).toArray().map((r) => Object.values(r.toJSON()));
      const r = await runRecords(parseTest(await (await fetch(`/${file}`)).text()), rows, { file, log: (s) => window.report(s) });
      await c.close();
      t.passed += r.passed; t.failed += r.failed; t.skipped += r.skipped;
      if (r.failed) t.failedFiles.push(file);
      await window.report(`  ${r.failed ? '✗' : '✓'} ${file}: ${r.passed} passed, ${r.failed} failed, ${r.skipped} skipped`);
    }
    return t;
  }, { files, platform });
  process.stdout.write(`Totals: ${totals.passed} passed, ${totals.failed} failed, ${totals.skipped} skipped\n`);
  if (totals.failed) { failed = true; process.stdout.write(`✗ failing: ${totals.failedFiles.join(', ')}\n`); }
} catch (e) {
  failed = true;
  process.stdout.write(`✗ HARNESS ERROR: ${e && e.stack ? e.stack : e}\n`);
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
