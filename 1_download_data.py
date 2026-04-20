"""
Step 1: Download Stock Data
Downloads historical OHLCV data for a list of tickers using yfinance
and saves them as CSV files in the ./data/ directory.
"""

import yfinance as yf
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS   = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
START     = "2020-01-01"
END       = "2024-12-31"
DATA_DIR  = "./data"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)

all_frames = []

for ticker in TICKERS:
    print(f"  Downloading {ticker} ...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

    if df.empty:
        print(f"    No data for {ticker}, skipping.")
        continue

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2 quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.rename(columns={
        "open":   "open_price",
        "high":   "high_price",
        "low":    "low_price",
        "close":  "close_price",
        "volume": "volume",
    }, inplace=True)

    df["symbol"] = ticker
    df = df[["symbol", "date", "open_price", "high_price",
             "low_price", "close_price", "volume"]]

    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(csv_path, index=False)
    all_frames.append(df)
    print(f"    ✓  {len(df)} rows → {csv_path}")

# Also write a combined CSV (handy for bulk-loading)
if all_frames:
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(DATA_DIR, "all_stocks.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n  Combined CSV: {combined_path}  ({len(combined)} total rows)")
else:
    print("  No data downloaded.")
