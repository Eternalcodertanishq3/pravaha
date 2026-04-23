# Contributing

## Development Setup

```bash
git clone https://github.com/pravaha/pravaha.git
cd pravaha
pip install -e ".[dev]"
```

## Code Standards

- **Python 3.11+** with modern syntax
- **Type hints** on all function signatures
- **Google-style docstrings** on all classes and public methods
- **Ruff** for linting: `ruff check pravaha/`
- **Mypy** for type checking: `mypy pravaha/ --ignore-missing-imports`

## Testing

```bash
pytest tests/ -v --cov=pravaha
```

Tests must pass before merging.

## Architecture

One file per agent. One class per file.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full system design.

## Pull Request Guidelines

1. Create a feature branch from `main`
2. Follow the code standards above
3. Add tests for new functionality
4. Update docs if adding new features
5. Run `ruff check` and `mypy` before submitting
