# YFinance MCP Server + VS Code (MCP Client) — Demo Guide

Audience: Microsoft EPS Cloud Solutions Architects
Author: Arturo Quiroga — Sr Azure AI Services Engineer & Solutions Architect
Date: 2025-08-23

This guide documents what we set up so you can reproduce a demo showing VS Code acting as an MCP client to a local yfinance MCP server, pulling Yahoo Finance market data for trading signal workflows.

---

## What you will demo

- VS Code (Copilot) as an MCP client
- A local yfinance MCP server (from PyPI) over STDIO
- Calling tools like get_stock_history, dividends, recommendations for tickers

Architecture (ASCII)

```
VS Code + Copilot (MCP client)
           │  (JSON-RPC over stdio)
           ▼
   yfinance-mcp-server (FastMCP) ────► yfinance (Yahoo Finance)
## Live harness (optional)

If you want to show CSVs being produced live from the MCP tools:

Setup once:

```bash
# Install harness deps (in your project venv or globally)
pip install -r scripts/requirements.txt

# Ensure the server is installed
pipx install yfinance-mcp-server
```

Run (stdio, spawns its own server instance):

```bash
python scripts/mcp_yfinance_dump.py \
  --symbols AAPL,MSFT \
  --period 1mo \
  --interval 1d \
  --outdir data
```

Outputs:
- `data/AAPL_history.csv`, `data/AAPL_dividends.csv`, `data/AAPL_features.csv`, etc.

Notes:
- If you use uv, run with: `python scripts/mcp_yfinance_dump.py --command uvx -- args yfinance-mcp-server ...`
- This script is separate from VS Code; both can run simultaneously since each starts its own server process.
---

## Prerequisites

- macOS (zsh shell)
- Visual Studio Code 1.102+ with GitHub Copilot access
- Internet access
- Homebrew and pipx (or uv) available
- This repo cloned locally: `~/GITHUB/TRADE-AI`

Check versions (optional):

```bash
code --version
brew --version
python3 --version
```

---

## 1) Install the yfinance MCP server

We use pipx to keep tools isolated and on PATH.

```bash
# Install pipx if needed
brew install pipx
pipx ensurepath
exec zsh

# Install the server from PyPI
pipx install yfinance-mcp-server
```

Smoke test (should print a FastMCP banner and wait on stdio):

```bash
pipx run yfinance-mcp-server
# Press Ctrl+C to exit
```

If you prefer uv instead of pipx:

```bash
# Install uv (if needed)
/bin/bash -c "$(curl -fsSL https://astral.sh/install.sh)"
export PATH="$HOME/.local/bin:$PATH"
uv tool install yfinance-mcp-server
```

---

## 2) VS Code MCP configuration

This repo contains a workspace-level MCP config at `.vscode/mcp.json` that tells VS Code to start the server via pipx over stdio.

File: `.vscode/mcp.json`

```json
{
  "inputs": [],
  "servers": {
    "yfinance": {
      "type": "stdio",
      "command": "pipx",
      "args": ["run", "yfinance-mcp-server"]
    }
  }
}
```

If you installed using uv, switch to:

```json
{
  "inputs": [],
  "servers": {
    "yfinance": {
      "type": "stdio",
      "command": "uvx",
      "args": ["yfinance-mcp-server"]
    }
  }
}
```

Notes:
- VS Code shows a trust prompt when starting a new MCP server—review and accept to proceed.
- MCP support is generally available in VS Code 1.102+.

Reference: VS Code MCP docs — https://code.visualstudio.com/docs/copilot/chat/mcp-servers

---

## 3) Run and verify in VS Code

1) Open this repo folder in VS Code.
2) Command Palette → “MCP: List Servers” and confirm `yfinance` appears.
3) If prompted, trust the server configuration.
4) Open Copilot Chat or Agent Mode (⌃⌘I).
5) In the Tools dropdown, enable the yfinance tools.
6) Ask Copilot to list tools or call one:

Examples to try in chat:
- “List the yfinance tools available.”
- “Fetch the daily OHLCV for AAPL for the last 30 days.”
- “Get dividends for MSFT for the past year.”
- “Show analyst recommendations for NVDA.”

Depending on the package version, typical tools include:
- `get_stock_history` (symbol, period, interval)
- `get_dividends`, `get_splits`, `get_recommendations`
- Financials, balance sheets, earnings calendar, options chain, etc.

If something fails, click “Show Output” in the Chat panel to view server logs.

---

## 4) Optional: HTTP mode and programmatic use

The package primarily runs over stdio for MCP clients. If it supports HTTP (per PyPI docs) you can run an HTTP endpoint and call it directly from Python:

```python
from fastmcp import FastMCPClient

client = FastMCPClient("http://localhost:8000")
for t in client.list_tools():
    print(t.name)

resp = client.call_tool(
    "get_stock_history",
    {"symbol": "AAPL", "period": "6mo", "interval": "1d"}
)
print(resp)
```

Consult the package README for the exact flag or snippet to serve HTTP if needed.

---

## Demo flow (suggested script)

1) Context: Explain MCP (client/server) and why stdio is simple/local.
2) Show `.vscode/mcp.json` and explain it points to `yfinance-mcp-server`.
3) Start a chat and list tools; highlight categories (prices, fundamentals, options, news).
4) Run a basic history query for AAPL; briefly discuss how you’d transform to features.
5) Pull dividends and recommendations to show breadth.
6) Mention optional HTTP mode and Python client for pipelines.

Timing: 7–10 minutes including Q&A.

---

## Troubleshooting

- Command not found: pipx
  - Install: `brew install pipx && pipx ensurepath && exec zsh`
- Server not visible in VS Code
  - Run “MCP: List Servers”; ensure `.vscode/mcp.json` exists in the open folder.
  - Try “Developer: Reload Window”.
- Server fails to start
  - Run manually: `pipx run yfinance-mcp-server` to see errors.
  - Ensure PATH includes pipx shims; `pipx ensurepath` then reopen the terminal/VS Code.
- Using uv instead
  - Change `command` to `uvx` in `.vscode/mcp.json` and ensure `~/.local/bin` is on PATH.
- Data issues (rate limiting/incomplete)
  - yfinance scrapes Yahoo and can be throttled—use smaller batches or add backoff.

---

## Uninstall / cleanup

```bash
# Remove the server
pipx uninstall yfinance-mcp-server

# Remove VS Code MCP config (if you want)
rm -f .vscode/mcp.json
```

---

## References

- PyPI: yfinance-mcp-server — https://pypi.org/project/yfinance-mcp-server/
- VS Code MCP servers — https://code.visualstudio.com/docs/copilot/chat/mcp-servers
- Model Context Protocol — https://modelcontextprotocol.io

---

## Appendix: Mapping to trading-signal workflows

- Use `get_stock_history` to retrieve OHLCV and compute indicators (SMA/EMA/RSI) offline.
- Use fundamentals (financials, balance sheet) for factor construction.
- Combine dividends, splits, actions for accurate total-return series.
- Batch pull multiple tickers on a cadence (Cron + HTTP mode) if desired.

This repo can be extended with scripts to call MCP tools and export CSV/Parquet; ask if you want a ready-made harness added.