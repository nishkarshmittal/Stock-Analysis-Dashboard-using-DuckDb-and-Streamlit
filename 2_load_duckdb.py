"""
Step 2: Load Data into DuckDB
Creates the DuckDB database file, defines the schema,
and bulk-loads all CSV files from ./data/.

DuckDB Internals Mapped Here
────────────────────────────
• CREATE TABLE uses columnar storage — each column is stored as a
  separate contiguous block, not row-by-row.
• COPY FROM CSV uses DuckDB's vectorized CSV reader; rows are ingested
  in batches (vectors of ~1 024 values) rather than one row at a time.
• The result is a .db file with column-wise layout ready for
  cache-friendly analytical scans.
"""

import duckdb
import os
import glob

DB_PATH  = "./stock_data.db"
DATA_DIR = "./data"

# ── Connect (creates the file if it doesn't exist) ────────────────────────────
con = duckdb.connect(DB_PATH)

# ── Schema ────────────────────────────────────────────────────────────────────
con.execute("""
    CREATE TABLE IF NOT EXISTS stock_data (
        symbol      VARCHAR,
        date        DATE,
        open_price  DOUBLE,
        high_price  DOUBLE,
        low_price   DOUBLE,
        close_price DOUBLE,
        volume      BIGINT
    )
""")

# ── Load CSVs ─────────────────────────────────────────────────────────────────
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
# Exclude the combined file so we don't double-count
csv_files = [f for f in csv_files if "all_stocks" not in os.path.basename(f)]

if not csv_files:
    # Fallback: load the combined file
    combined = os.path.join(DATA_DIR, "all_stocks.csv")
    if os.path.exists(combined):
        csv_files = [combined]
    else:
        raise FileNotFoundError(
            "No CSV files found in ./data/. Run 1_download_data.py first.")

# Clear existing data so re-runs are idempotent
con.execute("DELETE FROM stock_data")

total = 0
for path in sorted(csv_files):
    con.execute(f"""
        COPY stock_data
        FROM '{path}'
        (HEADER TRUE, DELIMITER ',')
    """)
    count = con.execute(
        "SELECT COUNT(*) FROM stock_data WHERE symbol = "
        f"(SELECT DISTINCT symbol FROM stock_data ORDER BY rowid DESC LIMIT 1)"
    ).fetchone()[0]
    print(f"  ✓  Loaded {os.path.basename(path)}")

total = con.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
print(f"\n✅  DuckDB ready: {DB_PATH}  ({total:,} rows)")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\nRow counts per symbol:")
result = con.execute("""
    SELECT symbol, COUNT(*) AS rows, MIN(date) AS earliest, MAX(date) AS latest
    FROM stock_data
    GROUP BY symbol
    ORDER BY symbol
""").df()
print(result.to_string(index=False))

con.close()
