# Contributing to nextaire_tools

Thanks for your interest in improving **nextaire_tools**! This guide covers the local
workflow, coding standards, and how to add new components.

## Development setup

```bash
git clone https://github.com/NextAIRE-Horizon/nextaire_tools.git
cd nextaire_tools
python -m venv .venv
# Windows:  .venv\Scripts\activate      Unix:  source .venv/bin/activate
pip install -e ".[dev,deep,extract,geo]"
```

## Quality gates

Every change must pass the same checks CI runs:

```bash
ruff check src tests        # lint
ruff format --check src tests
mypy src/nextaire_tools              # static types
pytest                      # tests (add --cov=nextaire_tools for coverage)
```

Run `ruff check --fix` and `ruff format` to auto-fix most issues.

## Coding standards

- Target Python 3.10+; every module starts with `from __future__ import annotations`.
- **Public API is typed and documented.** Full type hints and NumPy-style docstrings
  (Parameters / Returns / Raises / Examples) on every public class and function.
- **Preprocessing steps subclass `nextaire_tools.preprocessing.base.BaseStep`.** Implement the
  `_fit` / `_transform` / `_validate_params` hooks; never mutate the caller's
  DataFrame; store learned state in `trailing_underscore_` attributes.
- **Optional dependencies are gated.** Import heavy/optional packages (torch, cdsapi,
  xarray, geopandas, prophet, holidays) *inside* the function that needs them via
  `nextaire_tools.utils.validation.require(...)`. The library must import with only the core
  dependencies installed.
- No `print()` (use `nextaire_tools.utils.logging.get_logger`), no `plt.show()` in library
  code, and no network / file IO at import time.
- Visualizations follow the shared, colorblind-safe theme in `nextaire_tools.viz.style`.

## Adding a preprocessing step

1. Create `src/nextaire_tools/preprocessing/<name>.py` with a `BaseStep` subclass.
2. Register it in `nextaire_tools/preprocessing/pipeline.py::STEP_REGISTRY` (so `Pipeline.from_config`
   can build it) and export it from `nextaire_tools/preprocessing/__init__.py` and `nextaire_tools/__init__.py`.
3. Add tests in `tests/` and a usage example to the docs.

## Commit & PR conventions

- Keep commits focused; write imperative subject lines ("Add IQR outlier method").
- Update `CHANGELOG.md` under `[Unreleased]`.
- Ensure all quality gates pass before opening the PR.

By contributing you agree that your contributions are licensed under the MIT License.
