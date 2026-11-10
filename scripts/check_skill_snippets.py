#!/usr/bin/env python3
"""Check the Claude Code skills in plugins/anofox-ml/skills against the built
extension.

- Every ```sql block runs in a fresh in-memory database after
  scripts/skill_snippets/fixtures.sql, with ./build/release/duckdb. A block
  whose first line is `-- synopsis` is a signature sketch and is not run.
- Every `ml_*` name mentioned anywhere in a SKILL.md must be a registered
  function, and every kind named in a ml_*(...'Kind'...) call or in backticks
  must be a row of ml_estimators() (names in NOT_KINDS excepted).

Usage: python3 scripts/check_skill_snippets.py [SKILL.md ...]
Exit status: 0 clean, 1 on any failure, 2 without a built duckdb.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DUCKDB = ROOT / "build" / "release" / "duckdb"
FIXTURES = (ROOT / "scripts" / "skill_snippets" / "fixtures.sql").read_text()
SKILLS = ROOT / "plugins" / "anofox-ml" / "skills"
# CamelCase words in backticks that are not ml_estimators() rows: other
# libraries' names, and OrdinalEncoder (VARCHAR input, handled by the
# extension itself).
NOT_KINDS = {"GridSearchCV", "SelectFromModel", "RFE", "BLOB", "VARCHAR", "DOUBLE",
             "BIGINT", "INTEGER", "BOOLEAN", "STRUCT", "NULL", "NaN", "OrdinalEncoder",
             "BalancedAccuracy"}


def run(sql: str, *flags: str) -> tuple[bool, str]:
    p = subprocess.run([str(DUCKDB), "-bail", "-batch", *flags, ":memory:"], input=sql,
                       capture_output=True, text=True, timeout=600)
    return p.returncode == 0, (p.stdout if p.returncode == 0 else p.stderr or p.stdout).strip()


def main() -> int:
    if not DUCKDB.exists():
        print(f"{DUCKDB} is missing: build with `GEN=ninja make release` first")
        return 2
    ok, out = run("SELECT DISTINCT function_name FROM duckdb_functions() WHERE function_name LIKE 'ml_%';"
                  "SELECT kind FROM ml_estimators();", "-list", "-noheader")
    if not ok:
        print(out)
        return 2
    names = set(re.findall(r"\b\w+\b", out))
    files = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(SKILLS.glob("*/SKILL.md"))
    failures = passed = 0
    for f in files:
        text = f.read_text()
        rel = f.resolve().relative_to(ROOT)
        for name in sorted(set(re.findall(r"\bml_[a-z0-9_]+\b", text)) - names):
            print(f"FAIL {rel}: unknown function {name}")
            failures += 1
        kinds = set(re.findall(r"ml_\w+\(\s*'([A-Z]\w+)'", text))
        kinds |= set(re.findall(r"`([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+|[A-Z][a-z]+[A-Z]{2,}\w*)`", text))
        for kind in sorted(kinds - names - NOT_KINDS):
            print(f"FAIL {rel}: unknown estimator kind {kind}")
            failures += 1
        for m in re.finditer(r"```sql\n(.*?)```", text, re.S):
            block = m.group(1)
            line = text[: m.start()].count("\n") + 2
            if block.lstrip().lower().startswith("-- synopsis"):
                continue
            ok, out = run(FIXTURES + "\n" + block)
            if ok:
                passed += 1
            else:
                failures += 1
                print(f"FAIL {rel}:{line}: {out.splitlines()[0] if out else 'error'}")
    print(f"check_skill_snippets: {passed} blocks passed, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
