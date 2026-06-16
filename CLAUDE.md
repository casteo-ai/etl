# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Casteo.ai ETL service — fetches, cleans, stores, and (eventually) serves financial market data for Agentic AI and ML workloads. Currently covers three data sources: Binance spot, Deribit derivatives, and FRED US rates.

## Output of methods and functions

The key methods and functions must return pandas dataframes unless they are hidden or private. Do not 
return pandas series, only pandas dataframes, even if they have one column. Do not return numpy arrays unless 
required to do so.

## Unit tests

The framework for unit test is PyTest (pytest). For every method or function you build, you must also create the 
relevant unit tests. These unit tests are stored in the project's folder under .../tests and the structure 
inside this folder must match perfectly the structure in the .../src folder. For example, if you have a python code 
file in .../src/feeds/file.py you must create an equivalent unit test file with the same name and folder 
structure in tests, which in this case would be .../tests/feeds/test_file.py

## Environment

This project uses a Conda environment. Prefer installing dependencies using conda when possible:
```bash
conda install <package>
```
if not possible then use pip:
```bash
pip install <package>
```

### Output file format for structured data
When storing to disk information containing structured data, prefer CSV format.

Example of structured data:
- time series
- option price surface
- option volatility surface
- term structure of interest rates
- term structure of forward (futures) prices

The format you use must be consistent across all files:
- comma separated
- column headers must be present
- do not write to disk the index as an integer, the first column in the CSV must be somthing like timestamp, datetime, expiry date, strike, etc.

### Output file format for unstructured data
When storing unstructured data to disk prefer JSON file format.

Example of unstructured data:
- new article or their summary
- trade ideas or investment thesis
- descriptions in text format

You must not only store the information itself, but also relevant metadata to help with context.
For example of contextual data:
- uuid
- source
- url
- published_at
- collected_at
- topics
- entities
- geography

## Running tests

```bash
pytest tests/
pytest tests/path/to/test_file.py::TestClass::test_name   # single test
```

The `tests/` directory exists but is currently empty.

## Architecture

All feed clients live under `src/feeds/`. Each is a **stateful class** that wraps an external API, returns `pd.DataFrame`, and reads credentials exclusively from environment variables.

### BinanceClient (`src/feeds/binance.py`)

Wraps `ccxt.binance` for historical OHLCV klines with Binance's extended volume fields (taker-buy/sell split, quote volume). Key design:

- **Incremental CSV cache** — on first `fetch_ohlcv()` for a `(symbol, timeframe)` pair, any existing CSV is loaded into `_cache`. The last row is always dropped from cache (potentially incomplete candle) and the file is trimmed before the next write (`_csv_needs_trim`).
- **Gap-filling** — only the missing window portions (before and/or after the cached range) are fetched from the API. Prepend triggers a full CSV rewrite; append flushes every `_FLUSH_SIZE` (10 000) candles incrementally so a timeout doesn't lose progress.
- **Data dir** default: `/home/poucho/code/etl/data/binance/`; file pattern: `{SYMBOL}_{timeframe}.csv` (slashes in symbol become underscores).

Credentials (only for private/account endpoints):
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

### DeribitClient (`src/feeds/deribit.py`)

Wraps `ccxt.deribit` for options surfaces, forward curves, positions, and order placement. Key design:

- **Instrument routing** — BTC and ETH have inverse contracts (settled in their own coin). All other underlyings (SOL, XRP, AVAX, etc.) are USDC-linear. Pass any underlying currency to market-data methods; `_settlement_currency()` resolves the correct Deribit settlement currency automatically.
- **Public vs. private** — instrument listing, option surface, and forward curve are all public. Positions, balance, and order placement require credentials and call `_require_auth()`.
- **`fetch_repo_spreads()`** cross-calls `FedClient` to decompose futures basis into risk-free carry and residual repo spread (bps).

Credentials:
- `DERIBIT_API_KEY`
- `DERIBIT_API_SECRET`
- `DERIBIT_TESTNET` — set to `"1"` or `"true"` for testnet

### FedClient (`src/feeds/fred.py`)

Fetches USD interest rate pillars from FRED (SOFR overnight through 10-year CMT) and bootstraps a zero-coupon discount curve:

- T-bill tenors (SOFR, DGS1MO, DGS3MO, DGS6MO): exact zero-coupon via simple interest.
- Bond tenors (DGS1–DGS10): coupon stripping with log-linear interpolation for intermediate coupon dates.
- Results are **cached for 3 hours** (FRED data is only published daily).
- Interpolation beyond the last pillar uses flat zero-rate extrapolation.

Credential:
- `FRED_API_KEY`

### Cross-client interaction

`DeribitClient.fetch_repo_spreads()` imports and instantiates `FedClient` lazily if none is provided. This is the only cross-feed dependency today.

### Context-manager protocol

All three clients implement `__enter__` / `__exit__` and should be used with `with` when possible to ensure CCXT network resources are released cleanly.
