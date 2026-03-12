# Claude Code Guidelines

## Mandatory: Run CI checks before declaring a task complete

After every code change, run `/check` to execute all the same checks that GitHub Actions runs (lint, format, type check, tests, docs build, doc examples). **Do not tell the user the task is complete until every check passes.**

The `/check` skill walks through each step and tells you how to fix failures as they come up.

## Project structure

- `src/` — Python source (the `njit_wrappers` package)
- `tests/` — pytest unit tests
- `docs/` — MkDocs documentation with inline runnable code examples
- `setup.py` — builds the C++ extension; run `python setup.py build_ext --inplace` after any C++ changes

## CI checks (what `/check` runs)

> **Keep this table in sync with `.github/workflows/ci.yml`.** If you add, remove, or modify a
> CI step in the workflow file, update this table too — and vice versa. The two must always be
> consistent.

| Step | Command |
|------|---------|
| C++ format (clang-format) | `find src -name '*.cpp' -o -name '*.h' -o -name '*.cc' -o -name '*.cxx' \| xargs clang-format --dry-run --Werror` |
| Lint | `ruff check .` |
| Format | `ruff format --check .` |
| Type check | `PYTHONPATH=src pyright src/` |
| Tests | `PYTHONPATH=src pytest --cov --cov-report=xml` |
| Docs build | `mkdocs build --strict` |
| Doc examples | `PYTHONPATH=src pytest --markdown-docs docs/` |

## Code style

- Line length: 88 characters (ruff default)
- Pyright is set to `strict` mode; all public APIs need type annotations
- `src/njit_wrappers/_tensor.py` is excluded from type checking (uses private numba APIs with no stubs)
