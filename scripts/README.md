# MCP yfinance harness

A tiny, demo-friendly script that:
- launches the yfinance MCP server (stdio) via pipx
- calls tools to fetch OHLCV history and dividends
- writes CSVs under `data/`
  - history: `<SYMBOL>_history.csv`
  - dividends: `<SYMBOL>_dividends.csv`
  - features (SMA/EMA/RSI): `<SYMBOL>_features.csv`

## Setup

```bash
# Install harness deps (within your project venv or globally)
pip install -r scripts/requirements.txt

# Ensure server is installed and on PATH
pipx install yfinance-mcp-server
```

## Run (stdio, default)

```bash
python scripts/mcp_yfinance_dump.py \
  --symbols AAPL,MSFT \
  --period 1mo \
  --interval 1d \
  --outdir data
```

Output files:
- `data/AAPL_history.csv`, `data/AAPL_dividends.csv`, `data/AAPL_features.csv`, etc.

## Run (HTTP)
If you expose the server over HTTP:

```bash
python scripts/mcp_yfinance_dump.py \
  --transport http \
  --server-url http://localhost:8000 \
  --symbols AAPL,MSFT \
  --period 1mo \
  --interval 1d \
  --outdir data
```

## Notes
- If your environment uses `uvx` instead of `pipx`, add `--command uvx -- args yfinance-mcp-server`.
- The script tolerates slight variations in dividends tool names across package versions.
- Features require pandas; if not installed, the script will skip feature generation gracefully.