"""
Step 4: Streamlit Dashboard
A single-page analytical dashboard that queries DuckDB live on every
user interaction.  Each chart maps directly to a DuckDB internal feature.

Run:
    streamlit run 4_dashboard.py
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DB_PATH = "./stock_data.db"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analytics · DuckDB",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Price Analytical Dashboard")
st.caption("Powered by **DuckDB** — in-process columnar + vectorized execution")

# ── Shared connection (cached so it persists across reruns) ───────────────────
@st.cache_resource
def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

con = get_connection()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    symbols = con.execute(
        "SELECT DISTINCT symbol FROM stock_data ORDER BY symbol"
    ).fetchdf()["symbol"].tolist()

    ticker = st.selectbox("Ticker", symbols, index=0)

    date_range = con.execute("""
        SELECT MIN(date), MAX(date) FROM stock_data
    """).fetchone()
    start_date = st.date_input("Start date", value=date_range[0], min_value=date_range[0], max_value=date_range[1])
    end_date   = st.date_input("End date",   value=date_range[1], min_value=date_range[0], max_value=date_range[1])

    st.divider()
    st.markdown("### DuckDB Internals\nEach chart section documents which\ninternal feature is exercised.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Price + Moving Averages
# Internal: Window functions over partitioned columnar vectors
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("1 · Price & Moving Averages")
with st.expander("🔬 DuckDB Internals — Window Functions over Columnar Vectors"):
    st.markdown("""
**What the app does:** Computes 20-day and 50-day moving averages.

**What DuckDB does internally:**
- `AVG(...) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN N PRECEDING AND CURRENT ROW)` is a **window function**.
- DuckDB partitions the columnar `close_price` vector by `symbol`, sorts it by `date`, and slides the window across the sorted batch.
- Only `symbol`, `date`, and `close_price` columns are scanned from disk — all other columns are skipped (columnar I/O pruning).

**Why it matters:** For 5 years × 7 symbols × 1 024-value vector batches, the entire rolling calculation fits in L2 cache per batch, avoiding costly memory round-trips.
    """)

ma_df = con.execute(f"""
    SELECT
        date,
        close_price,
        ROUND(AVG(close_price) OVER (
            PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 2) AS ma_20,
        ROUND(AVG(close_price) OVER (
            PARTITION BY symbol ORDER BY date
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW), 2) AS ma_50
    FROM stock_data
    WHERE symbol = '{ticker}'
      AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
""").df()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=ma_df["date"], y=ma_df["close_price"],
                          name="Close", line=dict(color="#4C9BE8", width=1.2)))
fig1.add_trace(go.Scatter(x=ma_df["date"], y=ma_df["ma_20"],
                          name="MA-20", line=dict(color="#F4A460", width=1.5, dash="dot")))
fig1.add_trace(go.Scatter(x=ma_df["date"], y=ma_df["ma_50"],
                          name="MA-50", line=dict(color="#E87C4C", width=1.5, dash="dash")))
fig1.update_layout(height=350, margin=dict(t=20, b=20),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig1, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Daily Returns + Rolling Volatility
# Internal: LAG() window + vectorized STDDEV; multi-CTE flattened by optimizer
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("2 · Daily Returns & 20-Day Rolling Volatility")
with st.expander("🔬 DuckDB Internals — LAG, STDDEV, Multi-CTE Optimization"):
    st.markdown("""
**What the app does:** Computes daily log returns and a 20-day rolling standard deviation (volatility proxy).

**What DuckDB does internally:**
- `LAG(close_price)` is resolved in a single partitioned sort pass over the columnar vector.
- The outer `STDDEV(...) OVER (...)` window reuses the already-sorted partition — DuckDB does **not** re-sort.
- Multiple CTEs are **flattened** by the optimizer into a single pipeline: no intermediate materialisation to disk.

**Why it matters:** A naïve row-by-row loop in Python/Pandas would require iterating millions of rows; DuckDB's vectorized pipeline handles this in cache-resident batches.
    """)

vol_df = con.execute(f"""
    WITH returns AS (
        SELECT date,
               close_price,
               (close_price - LAG(close_price) OVER (ORDER BY date))
               / LAG(close_price) OVER (ORDER BY date) AS daily_return
        FROM stock_data
        WHERE symbol = '{ticker}'
          AND date BETWEEN '{start_date}' AND '{end_date}'
    )
    SELECT date,
           ROUND(daily_return * 100, 3)  AS daily_return_pct,
           ROUND(STDDEV(daily_return) OVER (
               ORDER BY date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * 100, 3) AS vol_20d
    FROM returns
    WHERE daily_return IS NOT NULL
    ORDER BY date
""").df()

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.5, 0.5], vertical_spacing=0.05)
fig2.add_trace(go.Bar(x=vol_df["date"], y=vol_df["daily_return_pct"],
                      name="Daily Return %",
                      marker_color=["#2ECC71" if v >= 0 else "#E74C3C"
                                    for v in vol_df["daily_return_pct"]]), row=1, col=1)
fig2.add_trace(go.Scatter(x=vol_df["date"], y=vol_df["vol_20d"],
                          name="Volatility 20d", line=dict(color="#9B59B6", width=1.5)),
               row=2, col=1)
fig2.update_layout(height=400, margin=dict(t=10, b=20),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RSI-14
# Internal: Multi-CTE pipeline, CASE vectorized via predicate masking
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("3 · RSI-14 Indicator")
with st.expander("🔬 DuckDB Internals — Multi-CTE Flattening & Predicate Masking"):
    st.markdown("""
**What the app does:** Computes the 14-period Relative Strength Index.

**What DuckDB does internally:**
- Three CTEs (`price_changes → gains_losses → avg_gl`) are **merged into one execution plan** by the query optimizer — no intermediate files.
- `CASE WHEN delta > 0 THEN delta ELSE 0 END` is executed as a **vectorized predicate mask** applied across an entire batch column at once (similar to numpy boolean indexing).

**Why it matters:** Without this optimization each CTE would materialize an intermediate result; DuckDB streams the data through all three stages in a single pass.
    """)

rsi_df = con.execute(f"""
    WITH price_changes AS (
        SELECT date,
               close_price,
               close_price - LAG(close_price) OVER (ORDER BY date) AS delta
        FROM stock_data
        WHERE symbol = '{ticker}'
          AND date BETWEEN '{start_date}' AND '{end_date}'
    ),
    gains_losses AS (
        SELECT date, close_price,
               CASE WHEN delta > 0 THEN delta ELSE 0 END AS gain,
               CASE WHEN delta < 0 THEN ABS(delta) ELSE 0 END AS loss
        FROM price_changes WHERE delta IS NOT NULL
    ),
    avg_gl AS (
        SELECT date, close_price,
               AVG(gain) OVER (ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
               AVG(loss) OVER (ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss
        FROM gains_losses
    )
    SELECT date,
           close_price,
           ROUND(CASE WHEN avg_loss = 0 THEN 100
                      ELSE 100 - (100 / (1 + avg_gain / NULLIF(avg_loss,0)))
                 END, 2) AS rsi_14
    FROM avg_gl ORDER BY date
""").df()

fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.6, 0.4], vertical_spacing=0.05)
fig3.add_trace(go.Scatter(x=rsi_df["date"], y=rsi_df["close_price"],
                          name="Close", line=dict(color="#4C9BE8")), row=1, col=1)
fig3.add_trace(go.Scatter(x=rsi_df["date"], y=rsi_df["rsi_14"],
                          name="RSI-14", line=dict(color="#E8A84C")), row=2, col=1)
fig3.add_hline(y=70, line_dash="dot", line_color="red",   row=2, col=1)
fig3.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
fig3.update_yaxes(range=[0, 100], row=2, col=1)
fig3.update_layout(height=420, margin=dict(t=10, b=20),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Volume Spike Detection
# Internal: Predicate Pushdown + columnar volume scan
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("4 · Volume Spike Detection")
with st.expander("🔬 DuckDB Internals — Predicate Pushdown"):
    st.markdown("""
**What the app does:** Highlights days where volume exceeded 2× the 30-day average — unusual trading activity.

**What DuckDB does internally:**
- `WHERE date BETWEEN ...` is a **predicate pushed down into the columnar scan layer**. Rows outside the date range are filtered before any aggregation operator sees them.
- With Parquet files DuckDB additionally uses **zone-map pruning** (block-level min/max metadata) to skip entire row groups without decompressing them.
- Only `date`, `symbol`, and `volume` columns are read — a 3-column scan on a 5-column table.

**Why it matters:** Predicate pushdown reduces I/O proportionally to how selective the filter is, often the single biggest performance lever on large datasets.
    """)

spike_df = con.execute(f"""
    WITH vol_avg AS (
        SELECT date, volume,
               AVG(volume) OVER (
                   ORDER BY date
                   ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS avg_vol_30d
        FROM stock_data
        WHERE symbol = '{ticker}'
          AND date BETWEEN '{start_date}' AND '{end_date}'
    )
    SELECT date, volume, ROUND(avg_vol_30d) AS avg_vol_30d,
           ROUND(volume / avg_vol_30d, 2) AS ratio
    FROM vol_avg ORDER BY date
""").df()

spike_df["spike"] = spike_df["ratio"] > 2.0

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=spike_df["date"], y=spike_df["volume"],
                      name="Volume",
                      marker_color=["#E74C3C" if s else "#7FB3D3"
                                    for s in spike_df["spike"]]))
fig4.add_trace(go.Scatter(x=spike_df["date"], y=spike_df["avg_vol_30d"],
                          name="30d Avg", line=dict(color="#F39C12", width=2)))
fig4.update_layout(height=320, margin=dict(t=10, b=20),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Cross-ticker Aggregation Comparison
# Internal: Columnar aggregation — GROUP BY on vectorized batches
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("5 · Cross-Ticker Aggregation Comparison")
with st.expander("🔬 DuckDB Internals — Vectorized GROUP BY Aggregation"):
    st.markdown("""
**What the app does:** Computes average close, total volume, and annualised volatility for all tickers simultaneously.

**What DuckDB does internally:**
- `GROUP BY symbol` uses a **hash aggregation** operator that processes data in columnar batches.
- `AVG`, `SUM`, `STDDEV` are all executed as **vectorized aggregation kernels** — the CPU processes 1 024 values per instruction cycle via SIMD.
- Compare this to MySQL's row-at-a-time tuple processing model where each aggregation step inspects one row before moving to the next.

**Why it matters:** For 5 years of data across 7 symbols, this single query replaces 7 separate Python loops and executes faster than a single Pandas `.groupby()`.
    """)

agg_df = con.execute(f"""
    WITH daily_returns AS (
        SELECT
            symbol,
            date,
            close_price,
            volume,
            (close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY date))
            / NULLIF(LAG(close_price) OVER (PARTITION BY symbol ORDER BY date), 0)
                AS daily_return
        FROM stock_data
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    )
    SELECT
        symbol,
        ROUND(AVG(close_price), 2)              AS avg_close,
        ROUND(MAX(close_price), 2)              AS max_close,
        ROUND(MIN(close_price), 2)              AS min_close,
        SUM(volume)                             AS total_volume,
        ROUND(STDDEV(daily_return) * SQRT(252) * 100, 2) AS annualised_vol_pct
    FROM daily_returns
    GROUP BY symbol
    ORDER BY avg_close DESC
""").df()

c1, c2 = st.columns(2)
with c1:
    fig5a = px.bar(agg_df, x="symbol", y="avg_close",
                   color="symbol", title="Avg Close Price",
                   color_discrete_sequence=px.colors.qualitative.Bold)
    fig5a.update_layout(height=320, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig5a, use_container_width=True)
with c2:
    fig5b = px.bar(agg_df, x="symbol", y="annualised_vol_pct",
                   color="symbol", title="Annualised Volatility (%)",
                   color_discrete_sequence=px.colors.qualitative.Bold)
    fig5b.update_layout(height=320, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig5b, use_container_width=True)

st.dataframe(agg_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "DSCI 551 · Nishkarsh Mittal · DuckDB in-process columnar OLAP engine · "
    "All queries run live against stock_data.db via DuckDB Python API"
)
