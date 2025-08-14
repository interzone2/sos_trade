import os
import io
import math
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

st.set_page_config(page_title="Show of Strength — Dashboard", layout="wide")
st.title("📈 Show of Strength — Signals Dashboard")
st.caption("Load your signals CSV (e.g., `show_of_strength_recent_7d.csv` or `show_of_strength_signals.csv`) to review entries, stops, and targets.")

@st.cache_data(show_spinner=False)
def load_csv(path_or_file) -> pd.DataFrame:
    if path_or_file is None:
        return pd.DataFrame()
    try:
        if isinstance(path_or_file, str):
            df = pd.read_csv(path_or_file)
        else:
            df = pd.read_csv(path_or_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame()

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

    # If this file is already filtered to signals, assume True when missing
    if "signal_sos" not in df.columns:
        df["signal_sos"] = True

    # Numeric coercions for convenience
    num_cols = [
        "open","high","low","close","volume",
        "entry_price","stop_price","target_2r","target_3r",
        "risk_per_share","rr_multiple"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()

    return df

# Sidebar: choose source
st.sidebar.header("Data")
default_path = st.sidebar.text_input(
    "CSV path (optional)",
    value="show_of_strength_recent_7d.csv",
    help="Point this at your signals file or upload below."
)
uploaded = st.sidebar.file_uploader("…or upload a signals CSV", type=["csv"])

df = None
if uploaded is not None:
    df = load_csv(uploaded)
elif default_path and os.path.exists(default_path):
    df = load_csv(default_path)
else:
    st.info("Provide a CSV path in the sidebar or upload a file to begin.")
    st.stop()

if df.empty:
    st.warning("No rows in the provided CSV.")
    st.stop()

# Ensure expected columns exist (we'll fill missing with NaNs/bools)
cols_main = [
    "date","ticker","signal_sos",
    "entry_price","stop_price","target_2r","target_3r",
    "risk_per_share","rr_multiple","entry_policy","stop_method","valid_trade"
]
for c in cols_main:
    if c not in df.columns:
        if c in ["signal_sos","valid_trade"]:
            df[c] = False
        else:
            df[c] = np.nan

# Derived helpers
if {"entry_price","stop_price"}.issubset(df.columns):
    if "risk_per_share" not in df.columns or df["risk_per_share"].isna().all():
        df["risk_per_share"] = (df["entry_price"] - df["stop_price"])
    df["risk_pct"] = (df["entry_price"] - df["stop_price"]) / df["entry_price"]
if {"target_2r","entry_price"}.issubset(df.columns):
    df["upside_to_2r_pct"] = (df["target_2r"] / df["entry_price"] - 1.0)

# Sidebar filters
st.sidebar.header("Filters")
dmin, dmax = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    (
        dmin.date() if pd.notnull(dmin) else date.today(),
        dmax.date() if pd.notnull(dmax) else date.today(),
    )
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = (dmin.date(), dmax.date())

tickers = sorted(df["ticker"].dropna().unique().tolist())
pick_tickers = st.sidebar.multiselect("Tickers", options=tickers, default=[])

min_rr = st.sidebar.number_input("Min RR multiple (2R default)", min_value=0.0, step=0.5, value=0.0)
must_be_valid = st.sidebar.checkbox("Valid trades only", value=True)

# Optional position sizing
st.sidebar.header("Position Sizing (optional)")
risk_usd = st.sidebar.number_input(
    "Account risk per idea (USD)",
    min_value=0.0, value=0.0, step=100.0,
    help="Leave 0 to skip sizing."
)

# Apply filters
mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
if pick_tickers:
    mask &= df["ticker"].isin(pick_tickers)
if must_be_valid and "valid_trade" in df.columns:
    mask &= df["valid_trade"].fillna(False)
if min_rr > 0 and "rr_multiple" in df.columns:
    mask &= df["rr_multiple"].fillna(0) >= min_rr

view = df.loc[mask, :].copy()

# Position sizing
if risk_usd > 0 and {"entry_price","stop_price"}.issubset(view.columns):
    view["risk_per_share"] = (view["entry_price"] - view["stop_price"])
    view["shares"] = np.where(
        view["risk_per_share"] > 0,
        np.floor(risk_usd / view["risk_per_share"]),
        np.nan
    )
    view["notional_usd"] = view["shares"] * view["entry_price"]
else:
    if "shares" not in view.columns:
        view["shares"] = np.nan
    if "notional_usd" not in view.columns:
        view["notional_usd"] = np.nan

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Rows", f"{len(view):,}")
with c2: st.metric("Unique tickers", f"{view['ticker'].nunique()}")
with c3: st.metric("Median Risk/Share", f"{np.nanmedian(view['risk_per_share']) if 'risk_per_share' in view.columns else float('nan'):.2f}")
with c4: st.metric("Median RR multiple", f"{np.nanmedian(view['rr_multiple']) if 'rr_multiple' in view.columns else float('nan'):.2f}")
with c5: st.metric("Valid trades", f"{int(view.get('valid_trade', pd.Series(dtype=bool)).sum())}")

# Main table
st.subheader("Signals")
show_cols = [
    "date","ticker","signal_sos",
    "entry_price","stop_price","risk_per_share","rr_multiple",
    "target_2r","target_3r","entry_policy","stop_method",
    "shares","notional_usd"
]
existing_cols = [c for c in show_cols if c in view.columns]
st.dataframe(view[existing_cols].sort_values(["date","ticker"]), use_container_width=True)

# Per-date counts
st.subheader("Counts by Date")
by_date = view.groupby(view["date"].dt.date).size().rename("count").reset_index()
st.dataframe(by_date, use_container_width=True)

# Export filtered view
st.subheader("Export")
csv_buf = io.StringIO()
view.to_csv(csv_buf, index=False)
st.download_button(
    "Download filtered CSV",
    data=csv_buf.getvalue(),
    file_name="sos_signals_filtered.csv",
    mime="text/csv"
)

st.caption("Tip: run the scan with `--recent-only` for a compact last-7-days file, or point this at `show_of_strength_signals.csv` for full history.")
