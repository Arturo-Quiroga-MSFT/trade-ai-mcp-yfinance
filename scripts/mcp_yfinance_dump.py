#!/usr/bin/env python3
"""
Tiny harness to call yfinance MCP tools and write CSVs for a quick demo.

Defaults to spawning the server via stdio using pipx:
  pipx run yfinance-mcp-server

Usage (examples):
  python scripts/mcp_yfinance_dump.py --symbols AAPL,MSFT --period 1mo --interval 1d --outdir data

Options:
  --transport stdio|http   Transport to use (default: stdio)
  --server-url URL         HTTP base URL if using http transport
  --command CMD            Override command for stdio (default: pipx)
  --args ARGS...           Override args for stdio (default: run yfinance-mcp-server)

This script tries get_stock_history and dividends tools and writes:
  data/<SYMBOL>_history.csv
  data/<SYMBOL>_dividends.csv

It is resilient to slight differences in tool names across versions by
trying multiple known variants for dividends.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional
import io

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # Optional; only needed for feature engineering


def coerce_rows(value: Any) -> List[Dict[str, Any]]:
    """Coerce arbitrary tool return shapes into a list of dict rows.

    Supports common shapes:
    - list[dict]
    - {"data": [...]}
    - {"rows": [...], "columns": [...]} -> reconstruct dicts
    - {"items": [...]} (fallback)
    - plain dict -> wrap in single row
    """
    if value is None:
        return []
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return value  # list of dicts
        # list of scalars -> make dicts with generic column
        return [{"value": v} for v in value]
    if isinstance(value, dict):
        # Content-item fallback: {"type":"text", "text": "...json or csv..."}
        if "type" in value and "text" in value and isinstance(value.get("text"), str):
            t = value["text"]
            # Try JSON first, then CSV
            try:
                parsed = json.loads(t)
                return coerce_rows(parsed)
            except Exception:
                rows = parse_csv_text(t)
                if rows:
                    return rows
                # If still not parseable, wrap original dict
                return [value]
        # Common wrappers
        if "data" in value and isinstance(value["data"], list):
            return coerce_rows(value["data"])
        if "records" in value and isinstance(value["records"], list):
            return coerce_rows(value["records"])
        if "items" in value and isinstance(value["items"], list):
            return coerce_rows(value["items"])
        # Table shape: rows + columns
        if (
            "rows" in value
            and isinstance(value["rows"], list)
            and "columns" in value
            and isinstance(value["columns"], (list, tuple))
        ):
            cols = list(value["columns"])
            rows = []
            for r in value["rows"]:
                if isinstance(r, dict):
                    rows.append(r)
                elif isinstance(r, (list, tuple)):
                    rows.append({c: r[i] if i < len(r) else None for i, c in enumerate(cols)})
                else:
                    rows.append({"value": r})
            return rows
        # Pandas "columns" orientation: {col: {index: val, ...}, ...}
        if value and all(isinstance(v, dict) for v in value.values()):
            # Build union of all inner keys as row indices
            inner_keys = set()
            for v in value.values():
                inner_keys.update(v.keys())
            rows = []
            for idx in sorted(inner_keys):
                row = {k: value[k].get(idx) if isinstance(value[k], dict) else None for k in value.keys()}
                # If index looks like a date, add explicit column
                row_key = str(idx)
                if any(ch.isdigit() for ch in row_key) and ("-" in row_key or "/" in row_key):
                    row.setdefault("date", row_key)
                else:
                    row.setdefault("index", idx)
                rows.append(row)
            return rows
        # Date-keyed orientation:
        # - values as dict: {"2025-01-01": {col: val, ...}, ...}
        # - values as scalar: {"2025-01-01": 0.25, ...}
        if value and all(isinstance(k, str) for k in value.keys()):
            looks_like_dates = 0
            for k in value.keys():
                if any(ch.isdigit() for ch in k) and ("-" in k or "/" in k):
                    looks_like_dates += 1
            if looks_like_dates >= max(1, len(value) // 2):
                # values dict -> merge; values scalar -> use 'value' column
                rows = []
                sample_val = next(iter(value.values())) if value else None
                if isinstance(sample_val, dict):
                    for date_key, cols in value.items():
                        if isinstance(cols, dict):
                            row = {**cols, "date": date_key}
                            rows.append(row)
                else:
                    for date_key, scalar in value.items():
                        if not isinstance(scalar, (list, dict)):
                            rows.append({"date": date_key, "value": scalar})
                if rows:
                    return rows
        # plain dict -> single row fallback
        return [value]
    # scalar
    return [{"value": value}]


def parse_csv_text(text: str) -> List[Dict[str, Any]]:
    """Parse CSV text into list of dict rows if it looks like CSV."""
    try:
        # Heuristic: must contain at least one newline and a comma in header
        if "\n" not in text or "," not in text.split("\n", 1)[0]:
            return []
        buf = io.StringIO(text)
        reader = csv.DictReader(buf)
        rows = [dict(r) for r in reader]
        return rows
    except Exception:
        return []


def extract_data_from_content(content: Any) -> Any:
    """Extract structured data from MCP content variants.

    Handles both 'modelcontextprotocol' and 'mcp' SDK response shapes:
    - content as list of items with json/data/text
    - plain dict/list
    - text that may be JSON or CSV
    """
    # If content is already rows-like, return it
    if isinstance(content, (list, dict)):
        # If it's a list of content items (dicts) with json/data/text,
        # attempt to unwrap the first relevant item.
        if isinstance(content, list) and content:
            for item in content:
                # dict-like
                if isinstance(item, dict):
                    if "json" in item:
                        return item["json"]
                    if "data" in item:
                        return item["data"]
                    if "text" in item:
                        t = item["text"]
                        try:
                            return json.loads(t)
                        except Exception:
                            rows = parse_csv_text(t)
                            if rows:
                                return rows
                            return t
                else:
                    # object-like with attributes
                    # Prefer direct attributes first
                    if hasattr(item, "text"):
                        try:
                            val = getattr(item, "text")
                            t = val() if callable(val) else val
                            try:
                                return json.loads(t)
                            except Exception:
                                rows = parse_csv_text(t)
                                if rows:
                                    return rows
                                return t
                        except Exception:
                            pass
                    if hasattr(item, "data"):
                        try:
                            val = getattr(item, "data")
                            return val() if callable(val) else val
                        except Exception:
                            pass
                    if hasattr(item, "json"):
                        try:
                            # Prefer model_dump_json (Pydantic v2) if available
                            if hasattr(item, "model_dump_json"):
                                raw_json = item.model_dump_json()
                            else:
                                val = getattr(item, "json")
                                raw_json = val() if callable(val) else val
                            # Parse the JSON string of the content item
                            try:
                                d = json.loads(raw_json)
                                if isinstance(d, dict):
                                    if "json" in d:
                                        return d["json"]
                                    if "data" in d:
                                        return d["data"]
                                    if "text" in d:
                                        t = d["text"]
                                        try:
                                            return json.loads(t)
                                        except Exception:
                                            rows = parse_csv_text(t)
                                            if rows:
                                                return rows
                                            return t
                                return d
                            except Exception:
                                # If the JSON string cannot be parsed, return as-is
                                return raw_json
                        except Exception:
                            pass
            return content
        # If it's a single dict that looks like a content item
        if isinstance(content, dict):
            if "json" in content:
                return content["json"]
            if "data" in content:
                return content["data"]
            if "text" in content:
                t = content["text"]
                try:
                    return json.loads(t)
                except Exception:
                    rows = parse_csv_text(t)
                    if rows:
                        return rows
                    return t
        return content
    # Scalar string: try json -> csv -> scalar
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            rows = parse_csv_text(content)
            if rows:
                return rows
            return content
    return content


def rename_value_column(rows: List[Dict[str, Any]], new_name: str) -> List[Dict[str, Any]]:
    """Rename a 'value' column to a more meaningful name if present."""
    if not rows:
        return rows
    # Only rename if 'value' exists and the new name doesn't already exist
    has_value = any("value" in r for r in rows)
    if not has_value:
        return rows
    # Create new list with renamed keys
    renamed: List[Dict[str, Any]] = []
    for r in rows:
        if "value" in r and new_name not in r:
            new_r = {**r}
            new_r[new_name] = new_r.pop("value")
            renamed.append(new_r)
        else:
            renamed.append(r)
    return renamed


def to_dataframe(value: Any, kind: Optional[str] = None) -> Optional[Any]:
    """Normalize arbitrary tool output into a pandas DataFrame.

    kind can be 'history' or 'dividends' to apply light conventions (e.g., rename value->dividend).
    Returns None if pandas isn't available or normalization fails.
    """
    if pd is None:
        return None

    def looks_like_date(s: str) -> bool:
        return any(ch.isdigit() for ch in s) and ("-" in s or "/" in s or "T" in s)

    obj = extract_data_from_content(value)

    def into_df(obj_any: Any) -> Any:
        # List cases
        if isinstance(obj_any, list):
            if obj_any and isinstance(obj_any[0], dict):
                return pd.DataFrame(obj_any)
            return pd.DataFrame({"value": obj_any})
        # Dict cases
        if isinstance(obj_any, dict):
            # unwrap common wrappers
            if "data" in obj_any and isinstance(obj_any["data"], list):
                return pd.DataFrame(obj_any["data"])
            if (
                "rows" in obj_any and isinstance(obj_any["rows"], list)
                and "columns" in obj_any and isinstance(obj_any["columns"], (list, tuple))
            ):
                cols = list(obj_any["columns"])
                rows = []
                for r in obj_any["rows"]:
                    if isinstance(r, dict):
                        rows.append(r)
                    elif isinstance(r, (list, tuple)):
                        rows.append({c: r[i] if i < len(r) else None for i, c in enumerate(cols)})
                    else:
                        rows.append({"value": r})
                return pd.DataFrame(rows)
            # pandas-like column orientation or arbitrary dict
            # Detect date-keyed mapping
            if obj_any and all(isinstance(k, str) for k in obj_any.keys()):
                date_keys = [k for k in obj_any.keys() if looks_like_date(k)]
                if len(date_keys) >= max(1, len(obj_any) // 2):
                    sample_val = next(iter(obj_any.values())) if obj_any else None
                    if isinstance(sample_val, dict):
                        rows = []
                        for dk, cols in obj_any.items():
                            if isinstance(cols, dict):
                                row = {**cols, "date": dk}
                                rows.append(row)
                        return pd.DataFrame(rows)
                    else:
                        # scalar series
                        rows = [{"date": dk, "value": v} for dk, v in obj_any.items()]
                        return pd.DataFrame(rows)
            # Fallback: flatten dict to single row
            try:
                return pd.json_normalize(obj_any)
            except Exception:
                return pd.DataFrame([obj_any])
        # String cases: try json, then CSV
        if isinstance(obj_any, str):
            try:
                parsed = json.loads(obj_any)
                return into_df(parsed)
            except Exception:
                try:
                    import io as _io
                    return pd.read_csv(_io.StringIO(obj_any))
                except Exception:
                    return pd.DataFrame([{ "value": obj_any }])
        # Scalar fallback
        return pd.DataFrame([{ "value": obj_any }])

    try:
        df = into_df(obj)
        # Light conventions by kind
        if kind == "dividends":
            if "value" in df.columns and "dividend" not in df.columns:
                df = df.rename(columns={"value": "dividend"})
        return df
    except Exception:
        return None


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        # Write an empty file with just a header marker for clarity
        with open(path, "w", newline="") as f:
            f.write("")
        return
    # Collect all keys across rows for a stable header
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def compute_features(history_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute simple indicators (SMA/EMA/RSI) from OHLCV history.

    Returns a list of dicts aligned to input rows with added feature columns.
    If pandas isn't available or close column cannot be found, returns an empty list.
    """
    if not history_rows or pd is None:
        return []
    try:
        df = pd.DataFrame(history_rows)
        # Try to locate a close price column
        close_col = None
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("close", "adj close", "adj_close", "adjclose"):
                close_col = c
                break
        if close_col is None:
            return []

        price = pd.to_numeric(df[close_col], errors="coerce")
        # Basic indicators
        df["sma_10"] = price.rolling(window=10, min_periods=1).mean()
        df["sma_20"] = price.rolling(window=20, min_periods=1).mean()
        df["sma_50"] = price.rolling(window=50, min_periods=1).mean()
        df["ema_12"] = price.ewm(span=12, adjust=False, min_periods=1).mean()
        df["ema_26"] = price.ewm(span=26, adjust=False, min_periods=1).mean()

        # RSI(14)
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Keep essential columns + features
        # If there is a Date/Datetime-like column, keep it first
        cols = list(df.columns)
        date_like = None
        for c in cols:
            cl = str(c).lower()
            if "date" in cl or "time" in cl:
                date_like = c
                break
        feature_cols = [
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi_14",
        ]
        keep = [date_like] if date_like else []
        keep += [close_col]
        keep += [c for c in feature_cols if c in df.columns]
        out = df[keep].copy()
        # Convert to list of dicts
        return out.to_dict(orient="records")
    except Exception:
        return []


async def run_stdio(symbols: List[str], period: str, interval: str, outdir: str, command: str, args: List[str]) -> None:
    """Connect over stdio using either modelcontextprotocol (preferred if installed)
    or fall back to the official 'mcp' SDK if not available.
    """
    import importlib

    using_legacy = False
    server = None
    session = None
    exit_stack = None

    # Helper adaptors to unify operations
    async def list_tools_adapter() -> List[str]:
        nonlocal using_legacy, server, session
        if using_legacy:
            resp = await server.list_tools()
            return [t.name for t in getattr(resp, "tools", [])]
        else:
            resp = await session.list_tools()
            return [t.name for t in getattr(resp, "tools", [])]

    async def call_tool_adapter(name: str, params: Dict[str, Any]) -> Any:
        nonlocal using_legacy, server, session
        if using_legacy:
            resp = await server.call_tool(name, params)
            raw = getattr(resp, "content", resp)
            return extract_data_from_content(raw)
        else:
            resp = await session.call_tool(name, params)
            # Try to unwrap 'mcp' SDK content into plain data
            content = getattr(resp, "content", resp)
            return extract_data_from_content(content)

    # Try legacy client first
    try:
        m = importlib.import_module("modelcontextprotocol.client.stdio")
        StdioServer = getattr(m, "StdioServer")
        server = await StdioServer.create(command=command, args=args)
        using_legacy = True
    except Exception:
        # Fallback to official 'mcp' SDK
        try:
            mcp = importlib.import_module("mcp")
            mcp_stdio = importlib.import_module("mcp.client.stdio")
            from contextlib import AsyncExitStack  # stdlib

            exit_stack = AsyncExitStack()
            # Build StdioServerParameters
            StdioServerParameters = getattr(mcp, "StdioServerParameters")
            params = StdioServerParameters(command=command, args=args, env=None)
            stdio_transport = await exit_stack.enter_async_context(getattr(mcp_stdio, "stdio_client")(params))
            reader, writer = stdio_transport
            ClientSession = getattr(mcp, "ClientSession")
            session = await exit_stack.enter_async_context(ClientSession(reader, writer))
            await session.initialize()
            using_legacy = False
        except Exception:
            print(
                "ERROR: No MCP Python client found. Install either 'modelcontextprotocol' (pip install -r scripts/requirements.txt) or the official 'mcp' SDK (pip install \"mcp[cli]\").",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        # Optional: list tools
        try:
            available = await list_tools_adapter()
            print("Available tools:", ", ".join(available) or "<none>")
        except Exception as e:
            print(f"Warning: failed to list tools: {e}")
            available = []

        for symbol in symbols:
            print(f"Processing {symbol} ...")
            # History
            history_data = await call_tool_adapter(
                "get_stock_history",
                {"symbol": symbol, "period": period, "interval": interval},
            )
            # Prefer pandas-based normalization
            df_hist = to_dataframe(history_data, kind="history")
            if df_hist is not None:
                df_hist.to_csv(os.path.join(outdir, f"{symbol}_history.csv"), index=False)
                history_rows = df_hist.to_dict(orient="records")
            else:
                history_rows = coerce_rows(history_data)
                write_csv(os.path.join(outdir, f"{symbol}_history.csv"), history_rows)
            print(f"  Wrote {len(history_rows)} rows -> {os.path.join(outdir, f'{symbol}_history.csv')}")

            # Compute features if possible
            features_rows = compute_features(history_rows)
            if features_rows:
                write_csv(os.path.join(outdir, f"{symbol}_features.csv"), features_rows)
                print(f"  Wrote features -> {os.path.join(outdir, f'{symbol}_features.csv')}")

            # Dividends (try several known tool names)
            dividends_rows: List[Dict[str, Any]] = []
            # Prefer a listed dividends tool to avoid validation warnings
            div_tool_candidates = []
            if 'available' in locals() and available:
                # rank by how specific the name is
                ranked = [
                    "get_dividends_and_splits",
                    "get_dividends_and_splits_history",
                    "get_dividends",
                ]
                for name in ranked:
                    if name in available:
                        div_tool_candidates.append(name)
            # Fallback order if none found in available
            if not div_tool_candidates:
                div_tool_candidates = [
                    "get_dividends_and_splits",
                    "get_dividends_and_splits_history",
                    "get_dividends",
                ]
            div_last_error: Optional[Exception] = None
            for tool_name in div_tool_candidates:
                try:
                    div_data = await call_tool_adapter(tool_name, {"symbol": symbol})
                    # If the combined tool returns both, try to isolate dividends
                    if isinstance(div_data, dict) and "dividends" in div_data:
                        dividends_rows = coerce_rows(div_data["dividends"])
                    else:
                        dividends_rows = coerce_rows(div_data)
                    if dividends_rows is not None:
                        break
                except Exception as e:
                    div_last_error = e
                    continue

            # Write dividends using pandas when possible
            if dividends_rows:
                if pd is not None:
                    df_div = to_dataframe(dividends_rows, kind="dividends")
                    if df_div is not None:
                        df_div.to_csv(os.path.join(outdir, f"{symbol}_dividends.csv"), index=False)
                        dividends_rows = df_div.to_dict(orient="records")
                    else:
                        dividends_rows = rename_value_column(dividends_rows, "dividend")
                        write_csv(os.path.join(outdir, f"{symbol}_dividends.csv"), dividends_rows)
                else:
                    dividends_rows = rename_value_column(dividends_rows, "dividend")
                    write_csv(os.path.join(outdir, f"{symbol}_dividends.csv"), dividends_rows)
            print(
                f"  Wrote {len(dividends_rows)} rows -> {os.path.join(outdir, f'{symbol}_dividends.csv')}"
                + (f" (last error: {div_last_error})" if div_last_error and not dividends_rows else "")
            )
    finally:
        try:
            if using_legacy and server is not None:
                await server.close()
            if not using_legacy and exit_stack is not None:
                await exit_stack.aclose()
        except Exception:
            pass


async def run_http(symbols: List[str], period: str, interval: str, outdir: str, server_url: str) -> None:
    try:
        import importlib
        fastmcp_mod = importlib.import_module("fastmcp")
        FastMCPClient = getattr(fastmcp_mod, "FastMCPClient")
    except Exception:
        print("ERROR: fastmcp not installed. Install it (pip install fastmcp) or use --transport stdio.", file=sys.stderr)
        sys.exit(1)

    client = FastMCPClient(server_url)

    # Optional: list tools
    try:
        for t in client.list_tools():
            print("Available tool:", t.name)
    except Exception as e:
        print(f"Warning: failed to list tools: {e}")

    for symbol in symbols:
        print(f"Processing {symbol} ...")
        # History
        history = client.call_tool(
            "get_stock_history",
            {"symbol": symbol, "period": period, "interval": interval},
        )
        df_hist = to_dataframe(history, kind="history")
        if df_hist is not None:
            df_hist.to_csv(os.path.join(outdir, f"{symbol}_history.csv"), index=False)
            history_rows = df_hist.to_dict(orient="records")
        else:
            history_rows = coerce_rows(extract_data_from_content(history))
            write_csv(os.path.join(outdir, f"{symbol}_history.csv"), history_rows)
        print(f"  Wrote {len(history_rows)} rows -> {os.path.join(outdir, f'{symbol}_history.csv')}")

        features_rows = compute_features(history_rows)
        if features_rows:
            write_csv(os.path.join(outdir, f"{symbol}_features.csv"), features_rows)
            print(f"  Wrote features -> {os.path.join(outdir, f'{symbol}_features.csv')}")

        # Dividends
        dividends_rows: List[Dict[str, Any]] = []
        div_tool_candidates = [
            "get_dividends",
            "get_dividends_and_splits",
            "get_dividends_and_splits_history",
        ]
        div_last_error: Optional[Exception] = None
        for tool_name in div_tool_candidates:
            try:
                div = client.call_tool(tool_name, {"symbol": symbol})
                div = extract_data_from_content(div)
                if isinstance(div, dict) and "dividends" in div:
                    # Prefer pandas for dividends
                    if pd is not None:
                        df_div = to_dataframe(div["dividends"], kind="dividends")
                        if df_div is not None:
                            df_div.to_csv(os.path.join(outdir, f"{symbol}_dividends.csv"), index=False)
                            dividends_rows = df_div.to_dict(orient="records")
                        else:
                            dividends_rows = coerce_rows(div["dividends"])
                    else:
                        dividends_rows = coerce_rows(div["dividends"])
                else:
                    if pd is not None:
                        df_div = to_dataframe(div, kind="dividends")
                        if df_div is not None:
                            df_div.to_csv(os.path.join(outdir, f"{symbol}_dividends.csv"), index=False)
                            dividends_rows = df_div.to_dict(orient="records")
                        else:
                            dividends_rows = coerce_rows(div)
                    else:
                        dividends_rows = coerce_rows(div)
                break
            except Exception as e:
                div_last_error = e
                continue

        if dividends_rows:
            if pd is None:
                dividends_rows = rename_value_column(dividends_rows, "dividend")
                write_csv(os.path.join(outdir, f"{symbol}_dividends.csv"), dividends_rows)
        print(
            f"  Wrote {len(dividends_rows)} rows -> {os.path.join(outdir, f'{symbol}_dividends.csv')}"
            + (f" (last error: {div_last_error})" if div_last_error and not dividends_rows else "")
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump history and dividends via yfinance MCP server")
    p.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g., AAPL,MSFT,NVDA")
    p.add_argument("--period", default="1mo", help="History period (e.g., 1mo, 6mo, 1y)")
    p.add_argument("--interval", default="1d", help="History interval (e.g., 1d, 1wk, 1mo)")
    p.add_argument("--outdir", default="data", help="Output directory for CSV files")
    p.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="MCP transport")
    p.add_argument("--server-url", default="http://localhost:8000", help="HTTP URL if transport=http")
    p.add_argument("--command", default="pipx", help="Command to launch server for stdio (default: pipx)")
    p.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=["run", "yfinance-mcp-server"],
        help="Args for the stdio command (default: run yfinance-mcp-server). Use -- to separate.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Prevent accidental capturing of our own --args in argparse
    if args.transport == "stdio":
        # If user didn't provide explicit args, argparse still sets the default
        if not args.args:
            args.args = ["run", "yfinance-mcp-server"]
        print(f"Starting MCP server via stdio: {args.command} {' '.join(args.args)}")
        asyncio.run(run_stdio(symbols, args.period, args.interval, args.outdir, args.command, args.args))
    else:
        print(f"Using HTTP transport at {args.server_url}")
        asyncio.run(run_http(symbols, args.period, args.interval, args.outdir, args.server_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
