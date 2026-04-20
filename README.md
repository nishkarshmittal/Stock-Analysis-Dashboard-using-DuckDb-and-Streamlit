# 📈 Stock Price Analytical Dashboard
### DSCI 551 — Database Internals Course Project | Nishkarsh Mittal

A stock analytics dashboard built with **DuckDB**, **Python**, and **Streamlit** that demonstrates how DuckDB's internal architecture (columnar storage, vectorized execution, predicate pushdown, window functions) directly powers real analytical workloads.

---

##  Project Structure

```
project/
├── data/                    # Downloaded CSV files (auto-created)
│   ├── AAPL.csv
│   ├── MSFT.csv
│   └── ...
├── stock_data.db            # DuckDB database file (auto-created)
├── 1_download_data.py       # Step 1 — Download OHLCV data via yfinance
├── 2_load_duckdb.py         # Step 2 — Load CSVs into DuckDB
├── 3_feature_queries.py     # Step 3 — Verify all 5 feature queries
├── 4_dashboard.py           # Step 4 — Streamlit dashboard
├── requirements.txt
└── README.md
```

---

##  Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9+** is required. Tested on Windows (VS Code terminal).

---

##  Run in Order

### Step 1 — Download stock data

```bash
python 1_download_data.py
```

Downloads historical OHLCV data for **AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META** from Yahoo Finance (2020–2024) and saves individual CSV files to `./data/`.

### Step 2 — Load into DuckDB

```bash
python 2_load_duckdb.py
```

Creates `stock_data.db` using DuckDB's columnar storage layout and bulk-loads all CSVs using DuckDB's vectorized CSV reader. Prints a per-symbol row count to confirm successful load.

### Step 3 — Verify feature queries *(optional but recommended)*

```bash
python 3_feature_queries.py
```

Runs all five analytical queries in the terminal with printed output. Use this to sanity-check the data and queries before launching the dashboard.

### Step 4 — Launch the dashboard

```bash
streamlit run 4_dashboard.py
```

Open **http://localhost:8501** in your browser.

---

##  Dashboard Sections & DuckDB Internals Mapping

Each section of the dashboard is explicitly linked to a DuckDB internal feature — this satisfies the DSCI 551 Phase 3 mapping requirement.

| # | Dashboard Section | DuckDB Internal Feature |
|---|---|---|
| 1 | Price + MA-20 / MA-50 | Window functions over partitioned columnar vectors |
| 2 | Daily Returns + Rolling Volatility | `LAG()` window + multi-CTE optimizer flattening |
| 3 | RSI-14 Indicator | Multi-CTE pipeline + vectorized `CASE` predicate masking |
| 4 | Volume Spike Detection | Predicate pushdown into the columnar scan layer |
| 5 | Cross-Ticker Aggregation | Vectorized `GROUP BY` hash aggregation (SIMD) |

Every section in the dashboard includes an expandable **"🔬 DuckDB Internals"** panel that explains:
- What the application does
- What DuckDB does internally
- Why that internal behavior matters for performance

---

##  Database Schema

```sql
CREATE TABLE stock_data (
    symbol      VARCHAR,
    date        DATE,
    open_price  DOUBLE,
    high_price  DOUBLE,
    low_price   DOUBLE,
    close_price DOUBLE,
    volume      BIGINT
);
```

---

##  DuckDB Internals Summary

| Concept | How It's Used in This Project |
|---|---|
| **Columnar Storage** | Only queried columns (e.g. `close_price`, `volume`) are read from disk; all others are skipped |
| **Vectorized Execution** | Aggregations like `AVG`, `STDDEV`, `SUM` process ~1,024 values per batch using SIMD |
| **Window Functions** | Moving averages, LAG returns, rolling volatility — all resolved over partitioned sorted vectors |
| **Predicate Pushdown** | `WHERE date BETWEEN ...` filters are pushed into the scan layer, skipping irrelevant rows before aggregation |
| **Multi-CTE Flattening** | The optimizer merges chains of CTEs into a single execution pipeline with no intermediate materialisation |

---

##  Dependencies

| Package | Purpose |
|---|---|
| `duckdb` | In-process analytical database engine |
| `yfinance` | Yahoo Finance data download |
| `pandas` | DataFrame handling and CSV I/O |
| `streamlit` | Interactive web dashboard |
| `plotly` | Interactive charts (candlestick, bar, line) |

---

##  Notes

- Re-running `2_load_duckdb.py` is safe — it clears existing rows before reloading (`DELETE FROM stock_data`).
- The dashboard uses a **cached read-only DuckDB connection** (`@st.cache_resource`) so queries are fast across Streamlit reruns.
- If yfinance returns a MultiIndex DataFrame (common in newer versions), `1_download_data.py` flattens it automatically.
