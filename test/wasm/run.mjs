#!/usr/bin/env node
// DuckDB-Wasm load + runtime harness for anofox_ml (after anofox-forecast's).
//
// Proves what the static import check cannot: the built .wasm LOADs in
// DuckDB-Wasm under Node and the functions give the test/sql results.
// Recipe: duckdb-node.cjs + web-worker, the eh / mvp bundle,
// allowUnsignedExtensions, and FORCE INSTALL from a localhost server that
// serves the artifact whatever version / platform path DuckDB-Wasm asks for.
//
// Usage: node run.mjs --ext <anofox_ml.duckdb_extension.wasm> [--bundle eh|mvp|base] [file.test ...]
// `base` is the bundle of the DuckDB 2.0 builds of @duckdb/duckdb-wasm (1.33.1-dev114.0).
// Without files it runs every test/sql/*.test except SKIP_FILES.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { parseTest, runRecords } from './sqllogic.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const require = createRequire(path.join(HERE, 'package.json'));
const EXT_NAME = 'anofox_ml';
const EXT_FILE = `${EXT_NAME}.duckdb_extension.wasm`;

// Files that cannot run on DuckDB-Wasm, with the reason (logged, never silent).
const SKIP_FILES = new Map([
  // Reads model BLOB fixtures from test/data with read_csv: no host filesystem in DuckDB-Wasm.
  ['test/sql/ml_blob_compat.test', 'reads test/data from the host filesystem'],
  // Not in the wasm builds: the perpetual crate builds a thread pool on every fit.
  ['test/sql/ml_perpetual.test', 'PerpetualBooster is not built for wasm (needs threads)'],
]);

function parseArgs(argv) {
  const args = { ext: process.env.ANOFOX_WASM_EXT || null, bundle: process.env.ANOFOX_WASM_BUNDLE || 'eh', files: [] };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--ext') args.ext = argv[++i];
    else if (a === '--bundle') args.bundle = argv[++i];
    else if (a.endsWith('.test')) args.files.push(a);
  }
  if (!args.ext) throw new Error('pass --ext <path to anofox_ml.duckdb_extension.wasm>');
  if (!['eh', 'mvp', 'base'].includes(args.bundle)) throw new Error(`--bundle must be eh, mvp or base, got ${args.bundle}`);
  return args;
}

function startServer(extPath) {
  const wasm = fs.readFileSync(extPath);
  const server = http.createServer((req, res) => {
    if (decodeURIComponent((req.url || '').split('?')[0]).endsWith(EXT_FILE)) {
      res.setHeader('Content-Type', 'application/wasm');
      res.end(wasm);
    } else {
      res.statusCode = 404;
      res.end('not found');
    }
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port })));
}

async function bootEngine(bundle) {
  const duckdb = require('@duckdb/duckdb-wasm/dist/duckdb-node.cjs');
  const Worker = require('web-worker');
  const dist = path.dirname(require.resolve('@duckdb/duckdb-wasm/dist/duckdb-node.cjs'));
  const worker = new Worker(path.join(dist, `duckdb-node-${bundle}.worker.cjs`));
  const db = new duckdb.AsyncDuckDB({ log() {} }, worker);
  await db.instantiate(path.join(dist, `duckdb-${bundle}.wasm`), null);
  await db.open({ allowUnsignedExtensions: true });
  return { db, worker };
}

const rowsOf = (conn) => async (sql) => (await conn.query(sql)).toArray().map((r) => Object.values(r.toJSON()));

async function main() {
  const args = parseArgs(process.argv);
  const log = (s) => process.stdout.write(s + '\n');
  const { server, port } = await startServer(path.resolve(args.ext));
  let engine;
  let failed = false;
  try {
    engine = await bootEngine(args.bundle);
    let conn = await engine.db.connect();
    log(`DuckDB-Wasm ${(await rowsOf(conn)('PRAGMA version'))[0][0]}, bundle ${args.bundle}`);
    await conn.query(`FORCE INSTALL ${EXT_NAME} FROM 'http://127.0.0.1:${port}'`);
    await conn.query(`LOAD ${EXT_NAME}`);
    log(`✓ LOAD ${EXT_NAME}: ${(await rowsOf(conn)('SELECT ml_version()'))[0][0]}`);
    await conn.close();

    const files = args.files.length
      ? args.files
      : fs.readdirSync(path.join(REPO_ROOT, 'test', 'sql')).filter((f) => f.endsWith('.test')).sort().map((f) => `test/sql/${f}`);
    let pass = 0, fail = 0, skip = 0;
    const failedFiles = [];
    for (const rel of files) {
      if (SKIP_FILES.has(rel)) { log(`  ⊘ ${rel}: skipped (${SKIP_FILES.get(rel)})`); continue; }
      // A fresh catalog per file, as sqllogictest does. DuckDB-Wasm 2.0
      // forgets the installed copy on re-open, so install it again.
      await engine.db.open({ allowUnsignedExtensions: true });
      conn = await engine.db.connect();
      await conn.query(`FORCE INSTALL ${EXT_NAME} FROM 'http://127.0.0.1:${port}'`);
      await conn.query(`LOAD ${EXT_NAME}`);
      const r = await runRecords(parseTest(fs.readFileSync(path.join(REPO_ROOT, rel), 'utf8')), rowsOf(conn), { file: rel, log });
      await conn.close();
      pass += r.passed; fail += r.failed; skip += r.skipped;
      log(`  ${r.failed ? '✗' : '✓'} ${rel}: ${r.passed} passed, ${r.failed} failed, ${r.skipped} skipped`);
      if (r.failed) failedFiles.push(rel);
    }
    log(`Totals: ${pass} passed, ${fail} failed, ${skip} skipped`);
    if (fail) { failed = true; log(`✗ failing: ${failedFiles.join(', ')}`); }
  } catch (err) {
    failed = true;
    log(`✗ HARNESS ERROR: ${err && err.stack ? err.stack : err}`);
  } finally {
    try { if (engine) { await engine.db.terminate(); engine.worker.terminate(); } } catch { /* ignore */ }
    server.close();
  }
  process.exit(failed ? 1 : 0);
}

main();
