Run all CI checks that GitHub Actions would run. Do not tell the user the task is complete until every check passes.

Run the following commands in order. If any command fails, fix the issue and re-run from that step before continuing.

**Step 1 — Lint (ruff)**
```
ruff check .
```
If it fails, run `ruff check --fix .` to auto-fix, then re-run to confirm clean.

**Step 2 — Format (ruff)**
```
ruff format --check .
```
If it fails, run `ruff format .` to auto-format, then re-run to confirm clean.

**Step 3 — Type check (pyright)**
```
PYTHONPATH=src pyright src/
```
Fix all type errors manually; pyright has no auto-fix.

**Step 4 — Unit tests (pytest)**
```
PYTHONPATH=src pytest --cov --cov-report=xml
```
Fix any failing tests before continuing.

**Step 5 — Build docs (mkdocs)**
```
mkdocs build --strict
```
Fix any doc build errors before continuing.

**Step 6 — Doc code examples (pytest-markdown-docs)**
```
PYTHONPATH=src pytest --markdown-docs docs/
```
Fix any failing doc examples before continuing.

Only after **all six steps pass** report to the user that the task is complete.
