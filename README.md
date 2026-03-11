# njit-wrappers

A hello world Python package using modern tooling.

## Toolchain

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Package manager & virtual env |
| [hatchling](https://hatch.pypa.io/) | Build backend |
| [ruff](https://docs.astral.sh/ruff/) | Linter & formatter |
| [pyright](https://github.com/microsoft/pyright) | Static type checker |
| [pytest](https://pytest.org/) | Test framework |

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate the virtual environment (optional, uv run handles this)
source .venv/bin/activate
```

## Usage

```python
from njit_wrappers import greet

print(greet())          # Hello, World!
print(greet("Python"))  # Hello, Python!
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Lint & format check
uv run ruff check .
uv run ruff format --check .

# Auto-fix lint issues and format
uv run ruff check --fix .
uv run ruff format .

# Type check
uv run pyright src/
```

## Project Structure

```
njit-wrappers/
├── pyproject.toml          # Project config & tool settings
├── src/
│   └── njit_wrappers/
│       ├── __init__.py
│       └── hello.py
├── tests/
│   ├── __init__.py
│   └── test_hello.py
└── README.md
```
