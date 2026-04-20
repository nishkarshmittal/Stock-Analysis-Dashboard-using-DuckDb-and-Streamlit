"""
Step 3: Feature Engineering Queries
Five SQL queries that compute analytical features directly inside DuckDB.
Each query is annotated with the DuckDB internals it exercises —
this is the "mapping" required by the DSCI 551 final report.

Run this script standalone to verify all queries work before the dashboard.
"""

import duckdb
import pandas as pd

DB_PATH = "./stock_data.db"
con = duckdb.connect(DB_PATH, read_only=True)

# ══════════════════════════════════════════════════════════════════════════════
# QUERY 1 — Simple Aggregation (Average Close Price per Symbol)
# ──────────────────────────────────────────────────────────────────────────────
# DuckDB Internals:
#   • Columnar scan: only the `symbol` and `close_price` columns are read from
#     disk.  All other columns are skipped entirely.
#   • Vectorized aggregation: AVG is computed over batches of ~1 024 values
#     rather than one row at a time, making full use of SIMD instructions.
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 60)
print("Q1  Average close price per symbol (columnar aggregation)")
q1 = con.execute("""
    SELECT
        symbol,
        ROUND(AVG(close_price), 2)  AS avg_close,
        ROUND(MIN(close_price), 2)  AS min_close,
        ROUND(MAX(close_price), 2)  AS max_close,
        COUNT(*)                    AS trading_days
    FROM stock_data
    GROUP BY symbol
    ORDER BY avg_close DESC
""").df()
print(q1.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 2 — 20-Day & 50-Day Moving Averages (Window Function)
# ──────────────────────────────────────────────────────────────────────────────
# DuckDB Internals:
#   • Window functions operate on sorted partitions of columnar vectors.
#   • PARTITION BY symbol keeps each ticker's data in a separate in-memory
#     segment; ORDER BY date ensures the rolling window is sequentially correct.
#   • Only close_price, symbol, date columns are scanned.
# ══════════════════════════════════════════════════════════════════════════════
print("\n─" * 60)
print("Q2  20-day and 50-day moving averages (window functions)")
q2 = con.execute("""
    SELECT
        symbol,
        date,
        close_price,
        ROUND(AVG(close_price) OVER (
            PARTITION BY symbol
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ), 2) AS ma_20,
        ROUND(AVG(close_price) OVER (
            PARTITION BY symbol
            ORDER BY date
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ), 2) AS ma_50
    FROM stock_data
    ORDER BY symbol, date
""").df()
print(q2.head(10).to_string(index=False))
print(f"  … {len(q2):,} total rows")


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 3 — Daily Returns & Rolling Volatility
# ──────────────────────────────────────────────────────────────────────────────
# DuckDB Internals:
#   • LAG() is a window function resolved over a partitioned, sorted vector.
#   • STDDEV over a rolling window uses a numerically stable vectorized
#     algorithm; the entire column lives in L2 cache for the batch size.
# ══════════════════════════════════════════════════════════════════════════════
print("\n─" * 60)
print("Q3  Daily returns & 20-day rolling volatility (LAG + STDDEV window)")
q3 = con.execute("""
    WITH returns AS (
        SELECT
            symbol,
            date,
            close_price,
            (close_price - LAG(close_price) OVER (
                PARTITION BY symbol ORDER BY date))
            / LAG(close_price) OVER (
                PARTITION BY symbol ORDER BY date)
            AS daily_return
        FROM stock_data
    )
    SELECT
        symbol,
        date,
        close_price,
        ROUND(daily_return * 100, 4)            AS daily_return_pct,
        ROUND(STDDEV(daily_return) OVER (
            PARTITION BY symbol
            ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) * 100, 4)                              AS volatility_20d
    FROM returns
    WHERE daily_return IS NOT NULL
    ORDER BY symbol, date
""").df()
print(q3.head(10).to_string(index=False))
print(f"  … {len(q3):,} total rows")


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 4 — Predicate Pushdown Filter + Volume Spike Detection
# ──────────────────────────────────────────────────────────────────────────────
# DuckDB Internals:
#   • The WHERE date > '2022-01-01' predicate is pushed down into the scan
#     operator — rows before 2022 are skipped before reaching the aggregation
#     layer.  With Parquet/columnar storage this also enables zone-map pruning
#     (block-level min/max metadata).
#   • Only date, symbol, volume columns are touched.
# ══════════════════════════════════════════════════════════════════════════════
print("\n─" * 60)
print("Q4  Volume spikes > 2× 30-day avg (predicate pushdown)")
q4 = con.execute("""
    WITH vol_avg AS (
        SELECT
            symbol,
            date,
            volume,
            AVG(volume) OVER (
                PARTITION BY symbol
                ORDER BY date
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) AS avg_vol_30d
        FROM stock_data
        WHERE date > '2022-01-01'        -- predicate pushed into columnar scan
    )
    SELECT
        symbol,
        date,
        volume,
        ROUND(avg_vol_30d)         AS avg_vol_30d,
        ROUND(volume / avg_vol_30d, 2) AS vol_ratio
    FROM vol_avg
    WHERE volume > 2 * avg_vol_30d
    ORDER BY vol_ratio DESC
    LIMIT 20
""").df()
print(q4.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# QUERY 5 — Relative Strength Index (RSI-14, pure SQL)
# ──────────────────────────────────────────────────────────────────────────────
# DuckDB Internals:
#   • Multiple CTEs are flattened by DuckDB's optimizer into a single
#     multi-pass columnar scan — no intermediate temp tables are written.
#   • The vectorized execution engine handles CASE-based branching across
#     entire batches via predicate masking (no per-row if/else branches).
# ══════════════════════════════════════════════════════════════════════════════
print("\n─" * 60)
print("Q5  RSI-14 indicator (multi-CTE vectorized execution)")
q5 = con.execute("""
    WITH price_changes AS (
        SELECT
            symbol, date, close_price,
            close_price - LAG(close_price) OVER (
                PARTITION BY symbol ORDER BY date) AS delta
        FROM stock_data
    ),
    gains_losses AS (
        SELECT
            symbol, date, close_price,
            CASE WHEN delta > 0 THEN delta ELSE 0 END AS gain,
            CASE WHEN delta < 0 THEN ABS(delta) ELSE 0 END AS loss
        FROM price_changes
        WHERE delta IS NOT NULL
    ),
    avg_gl AS (
        SELECT
            symbol, date, close_price,
            AVG(gain) OVER (
                PARTITION BY symbol
                ORDER BY date
                ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
            AVG(loss) OVER (
                PARTITION BY symbol
                ORDER BY date
                ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss
        FROM gains_losses
    )
    SELECT
        symbol,
        date,
        ROUND(close_price, 2)  AS close_price,
        ROUND(CASE
            WHEN avg_loss = 0 THEN 100
            ELSE 100 - (100 / (1 + avg_gain / NULLIF(avg_loss, 0)))
        END, 2)                AS rsi_14
    FROM avg_gl
    ORDER BY symbol, date
""").df()
print(q5.head(10).to_string(index=False))
print(f"  … {len(q5):,} total rows")

con.close()
print("\n✅  All five feature queries executed successfully.")
