# Trade AI — MCP yfinance demo

This repo shows how to use a yfinance MCP server from VS Code (MCP client), a small Python harness to dump data to CSVs and compute indicators, and a Plotly notebook to visualize signals.

## Overview
- VS Code MCP config: `.vscode/mcp.json` launches the yfinance MCP server over stdio.
- Data harness: `scripts/mcp_yfinance_dump.py` fetches history/dividends via MCP, writes readable CSVs, and computes SMA/EMA/RSI features with pandas.
- Notebook: `notebooks/plot_signals.ipynb` plots Close with SMA/EMA overlays, RSI, and MACD; handles inline/browser renderer fallbacks.
- Demo doc: `docs/yfinance-mcp-vscode-demo.md` has setup/troubleshooting and optional http mode.

## Quickstart

1) Install Python deps (for the harness + notebook)
```bash
pip install -r scripts/requirements.txt
```

2) Run the harness to fetch data and compute features
```bash
python scripts/mcp_yfinance_dump.py --symbols AAPL MSFT --mode stdio --out data
```
- Outputs: `data/<SYMBOL>_history.csv`, `data/<SYMBOL>_dividends.csv`, `data/<SYMBOL>_features.csv`

3) Open the notebook and plot
- Open `notebooks/plot_signals.ipynb` in VS Code and Run All. The first cell installs nbformat/ipython if missing and safely configures a Plotly renderer.

4) VS Code MCP server
- Ensure `.vscode/mcp.json` points to your yfinance MCP server (installed via pipx/uvx). Restart VS Code to auto-start the server.

## Notes
- The harness prefers pandas for normalization; row-based fallbacks are included.
- Date parsing is UTC-aware to avoid mixed-timezone warnings in future pandas versions.
- Plotly rendering falls back to the browser automatically if inline mime rendering isn’t available.

## License
- Add a LICENSE file if you plan to share or reuse this code. MIT is a common choice.