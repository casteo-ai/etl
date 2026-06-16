# Repository Guidelines

## Project Structure & Module Organization
This repository contains a Python ETL service for financial market data. Source code lives under `src/`, with feed clients in `src/feeds/`:

- `binance.py` for Binance spot market data and CSV caching.
- `deribit.py` for Deribit derivatives, positions, and repo spread calculations.
- `fred.py` for FRED rate data and discount-curve bootstrapping.

Tests live under `tests/` and mirror the source tree, for example `src/feeds/binance.py` is covered by `tests/feeds/test_binance.py`. Root-level `conftest.py` keeps the repository root importable as `src.*`. Runtime data is expected under `data/` or client-specific `data_dir` paths and should not be treated as source.

## Build, Test, and Development Commands
There is no package/build configuration yet, so work from the repository root and use pytest directly.

- `pytest tests/` runs the full test suite.
- `pytest tests/feeds/test_binance.py::TestToMs::test_iso_string` runs one focused test.
- `conda install <package>` is preferred for dependencies; use `pip install <package>` only when conda is unavailable.

Core dependencies used by the code and tests include `pandas`, `numpy`, `pytest`, `ccxt`, and FRED API support.

## Coding Style & Naming Conventions
Use Python 3 style with 4-space indentation, type hints where they clarify interfaces, and `from __future__ import annotations` in new modules for consistency. Keep feed clients as stateful classes that wrap external APIs. Public methods and functions should return `pandas.DataFrame` objects, not Series or NumPy arrays, unless there is a clear technical reason.

Name tests as `test_<module>.py`; use `Test<ClassOrArea>` classes and `test_<behavior>` methods. Keep helper functions private with a leading underscore when they are not part of the client API.

## Testing Guidelines
Use pytest. Add or update tests for every new public method or meaningful behavior change. Avoid live network calls in unit tests; mock CCXT, FRED, and filesystem interactions where practical. Use `tmp_path` for CSV persistence tests and `monkeypatch` for environment variables such as `BINANCE_API_KEY`, `DERIBIT_API_SECRET`, and `FRED_API_KEY`.

## Commit & Pull Request Guidelines
Git history currently contains only `Initial commit`, so no detailed commit convention is established. Use short, imperative commit subjects such as `Add Deribit forward curve tests`. Pull requests should include a concise summary, the tests run, any new environment variables, and notes about data files or external API behavior.

## Security & Configuration Tips
Read credentials from environment variables only. Do not commit API keys, generated market data, caches, or local IDE metadata. Prefer CSV for structured time series and JSON for unstructured records with metadata.
