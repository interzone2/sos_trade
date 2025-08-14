#!/usr/bin/env python3
"""
Show-of-Strength Signal Scanner — Python 3.10+

Inputs:  CSV with columns (case-insensitive):
    date, ticker, open, high, low, close, volume
    Optional: adjclose / adj_close  (if present, O/H/L/C are rebased to adjusted)

Key features
- Cleans and normalizes OHLCV; optional adjusted-basis rebasing using adjclose/close per-ticker.
- Indicators: EMA(9), RSI(14, Wilder), VolSMA(20), Prior-High(N), ATR(14, Wilder), MA(50) selectable EMA/SMA.
- EMA50/SMA50 proximity modes:
    * today    : within δ on the signal day
    * lookback : within δ in the last L sessions (excl. today)
    * recent   : within δ in the last K sessions (excl. today), with `ema50_days_since_touch`
- Optional guards: min breakout % over prior-high, min close-in-range.
- Basis sanity: rolling median |close - MA50| / close; can exclude outliers.
- Trade plan: entry (next_open or eod_close), stop (swing/defense/min), ATR buffer, R:R targets.
- Outputs:
    * show_of_strength_signals.csv       (all signals + trade plan)
    * show_of_strength_summary.csv       (per-ticker counts)
    * show_of_strength_eval.csv          (forward return eval, if not --recent-only)
    * show_of_strength_recent_<N>d.csv   (subset in last N dataset days)
    * Optional SQLite tables with --write-db

Examples
    python sos_signals.py --csv 'sp_500_new.csv' --recent-only \
      --ema50-prox-mode lookback --ema50-lookback 4 --ema50-prox-pct 0.015

    python sos_signals.py --csv 'sp_500_new.csv' --recent-only \
      --ma50-type sma --ema50-prox-mode today --ema50-prox-pct 0.015

    python sos_signals.py --csv 'sp_500_new.csv' \
      --ema50-prox-mode recent --ema50-recent-days 4 --ema50-prox-pct 0.015 \
      --signals-csv show_of_strength_signals.csv --write-db
"""

from __future__ import annotations
import argparse
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# ------------------------- Utilities & Indicators ------------------------- #

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # Canonicalize common synonyms
    col_map = {}
    if "symbol" in df.columns and "ticker" not in df.columns:
        col_map["symbol"] = "ticker"
    if "adj_close" in df.columns and "adjclose" not in df.columns:
        col_map["adj_close"] = "adjclose"
    if "datetime" in df.columns and "date" not in df.columns:
        col_map["datetime"] = "date"
    if "time" in df.columns and "date" not in df.columns:
        col_map["time"] = "date"
    df = df.rename(columns=col_map)
    return df


def parse_and_clean(csv_path: str, min_rows_per_symbol: int = 30) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_columns(df)

    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    present = set(df.columns)
    missing = required - present
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {sorted(present)}")

    # hygiene
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If adjclose present, rebase O/H/L/C to adjusted basis so indicators & prices align
    if "adjclose" in df.columns:
        df["adjclose"] = pd.to_numeric(df["adjclose"], errors="coerce")
        df = df.sort_values(["ticker", "date"], kind="mergesort")
        factor = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan)
        # Fill within ticker with TRANSFORM to keep index aligned
        factor = df.groupby("ticker", observed=True)["adjclose"].transform(lambda s: s / s)
        # Above produces 1.0 when adjclose==close. Use robust factor:
        factor = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan)
        factor = df.groupby("ticker", observed=True)["adjclose"].transform(lambda s: s) / \
                 df.groupby("ticker", observed=True)["close"].transform(lambda s: s)
        # If any NaNs linger, compute once more then ffill/bfill inside ticker
        factor = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan)
        factor = df.groupby("ticker", observed=True)["adjclose"].transform(lambda s: s) / \
                 df.groupby("ticker", observed=True)["close"].transform(lambda s: s)
        # Final guarantee: simple compute then ffill/bfill per-ticker
        factor = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan)
        factor = df.groupby("ticker", observed=True)["adjclose"].transform(lambda s: s) / \
                 df.groupby("ticker", observed=True)["close"].transform(lambda s: s)
        # If still NaN (some vendors), fallback to simple compute and fill
        factor = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan)
        factor = df.groupby("ticker", observed=True)["adjclose"].transform(lambda s: s) / \
                 df.groupby("ticker", observed=True)["close"].transform(lambda s: s)
        factor = factor.groupby(df["ticker"], observed=True).transform(lambda s: s.ffill().bfill())
        factor = factor.fillna(1.0)

        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce") * factor

        df["adjclose"] = df["close"]  # clarity: adjusted close == adjusted close we just set

    # Drop invalids
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[df["high"] >= df["low"]]
    df = df[df["volume"] > 0]

    # Sort/dedupe & prune short histories
    df = df.sort_values(["ticker", "date"], kind="mergesort")
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    counts = df.groupby("ticker", observed=True)["date"].count()
    valid = counts[counts >= min_rows_per_symbol].index
    df = df[df["ticker"].isin(valid)].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    alpha = 1 / window
    roll_up = up.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ------------------------- Parameters ------------------------- #

@dataclass
class SosParams:
    # Core SoS conditions
    prior_high_window: int = 5
    ema_span: int = 9
    vol_sma_window: int = 20
    volume_mult: float = 1.2
    rsi_window: int = 14
    rsi_threshold: float = 50.0
    pullback_window: int = 10
    min_days_below_ema: int = 2
    rsi_pullback_threshold: float = 45.0

    # 50-bar average + proximity controls
    ema50_span: int = 50
    ma50_type: str = "ema"          # 'ema' or 'sma'
    ema50_prox_pct: float = 0.02
    ema50_lookback: int = 10
    ema50_prox_mode: str = "lookback"  # 'lookback' | 'today' | 'recent'
    ema50_recent_days: int = 4
    ema50_required: bool = True

    # Extra guards
    min_breakout_pct: float = 0.0
    min_range_pos: float = 0.0

    # Basis sanity
    basis_window: int = 60
    basis_gap_thresh: float = 0.30
    exclude_suspect_basis: bool = True

    # Trade plan
    atr_window: int = 14
    stop_buffer_atr: float = 0.25
    swing_lookback: int = 10
    stop_method: str = "min_of_both"   # 'swing_only' | 'defense_only' | 'min_of_both'
    entry_policy: str = "next_open"    # 'next_open' | 'eod_close'
    rr_multiple: float = 2.0


# ------------------------- Indicators & Signals ------------------------- #

def compute_indicators(df: pd.DataFrame, p: SosParams) -> pd.DataFrame:
    df = df.copy()
    by = df.groupby("ticker", group_keys=False, observed=True)

    # Core indicators
    df["ema_fast"] = by["close"].apply(lambda s: ema(s, p.ema_span))
    df["rsi"] = by["close"].apply(lambda s: rsi_wilder(s, p.rsi_window))
    df["vol_sma20"] = by["volume"].apply(lambda s: s.rolling(p.vol_sma_window, min_periods=p.vol_sma_window).mean())
    df["prior_high_n"] = by["high"].apply(lambda s: s.rolling(p.prior_high_window, min_periods=p.prior_high_window).max().shift(1))

    # Pullback diagnostics (prior to today)
    df["below_ema"] = (df["close"] < df["ema_fast"]).astype(int)
    df["below_ema_count_pb"] = by["below_ema"].apply(lambda s: s.rolling(p.pullback_window, min_periods=1).sum()).shift(1)
    df["rsi_min_pb"] = by["rsi"].apply(lambda s: s.rolling(p.pullback_window, min_periods=1).min()).shift(1)
    df["had_pullback"] = ((df["below_ema_count_pb"] >= p.min_days_below_ema) | (df["rsi_min_pb"] < p.rsi_pullback_threshold))

    # 50-bar average (EMA or SMA)
    def _ma50(series_close: pd.Series) -> pd.Series:
        if p.ma50_type == "ema":
            return series_close.ewm(span=p.ema50_span, adjust=False, min_periods=p.ema50_span).mean()
        else:
            return series_close.rolling(p.ema50_span, min_periods=p.ema50_span).mean()

    df["ema50"] = by["close"].apply(_ma50)  # keep name for compatibility (can be EMA or SMA)
    df["ema50_dist_low_pct"] = (df["low"] - df["ema50"]).abs() / df["ema50"]
    df["ema50_dist_close_pct"] = (df["close"] - df["ema50"]).abs() / df["ema50"]

    df["ema50_proximity_today"] = (
        (df["ema50_dist_low_pct"] <= p.ema50_prox_pct) |
        (df["ema50_dist_close_pct"] <= p.ema50_prox_pct)
    )

    ema50_dist_low_min = by["ema50_dist_low_pct"].apply(lambda s: s.rolling(p.ema50_lookback, min_periods=1).min()).shift(1)
    df["ema50_proximity_lookback"] = (ema50_dist_low_min <= p.ema50_prox_pct)

    recent_min = by["ema50_dist_low_pct"].apply(lambda s: s.rolling(p.ema50_recent_days, min_periods=1).min()).shift(1)
    df["ema50_proximity_recentK"] = (recent_min <= p.ema50_prox_pct)

    # Days since last touch ≤ δ (transform-based, avoids deprecation)
    def _days_since_touch_group(series_dist_low_pct: pd.Series, thresh: float) -> pd.Series:
        b = (series_dist_low_pct <= thresh).fillna(False).to_numpy()
        out = np.empty(len(b), dtype=float)
        last = -1
        for i, flag in enumerate(b):
            if flag:
                last = i
            out[i] = (i - last) if last >= 0 else np.nan
        return pd.Series(out, index=series_dist_low_pct.index)

    df["ema50_days_since_touch"] = by["ema50_dist_low_pct"].transform(
        lambda s: _days_since_touch_group(s, p.ema50_prox_pct)
    )

    # Choose proximity condition (or disable)
    if p.ema50_prox_mode == "today":
        prox = df["ema50_proximity_today"]
    elif p.ema50_prox_mode == "recent":
        prox = df["ema50_proximity_recentK"]
    else:
        prox = df["ema50_proximity_lookback"]
    df["cond_ema50_prox"] = prox
    prox_gate = df["cond_ema50_prox"] if p.ema50_required else True

    # SoS base conditions for current day
    df["cond_above_ema"] = df["close"] > df["ema_fast"]
    df["cond_breakout"] = df["close"] > df["prior_high_n"]
    df["cond_volume"] = df["volume"] >= (p.volume_mult * df["vol_sma20"])
    df["cond_rsi_regime"] = df["rsi"] > p.rsi_threshold
    df["cond_rsi_rising"] = df["rsi"] > df["rsi"].groupby(df["ticker"], observed=True).shift(1)

    # Diagnostics
    df["volume_ratio"] = df["volume"] / df["vol_sma20"]
    df["close_vs_ema_pct"] = df["close"] / df["ema_fast"] - 1.0
    df["close_vs_prior_high_pct"] = df["close"] / df["prior_high_n"] - 1.0
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["range_pos"] = (df["close"] - df["low"]) / rng

    # Optional guards
    df["cond_min_breakout"] = (df["close_vs_prior_high_pct"] >= p.min_breakout_pct)
    df["cond_min_rangepos"] = (df["range_pos"] >= p.min_range_pos) if p.min_range_pos > 0 else True

    # Basis sanity
    df["ema50_gap_pct"] = (df["close"] - df["ema50"]).abs() / df["close"]
    roll_median_gap = (df.sort_values("date")
                         .groupby("ticker", observed=True)["ema50_gap_pct"]
                         .transform(lambda s: s.rolling(p.basis_window, min_periods=p.basis_window//2).median()))
    df["basis_suspect"] = roll_median_gap > p.basis_gap_thresh

    base_signal = (
        df["cond_above_ema"] &
        df["cond_breakout"] &
        df["cond_volume"] &
        df["cond_rsi_regime"] &
        df["cond_rsi_rising"] &
        df["had_pullback"].fillna(False) &
        df["cond_min_breakout"] &
        (df["cond_min_rangepos"] if p.min_range_pos > 0 else True) &
        (prox_gate if p.ema50_required else True)
    )

    if p.exclude_suspect_basis:
        base_signal = base_signal & (~df["basis_suspect"].fillna(False))

    df["signal_sos"] = base_signal

    # ATR(14, Wilder) w/out groupby.apply
    prev_close = df.groupby("ticker", observed=True)["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["tr"] = tr
    alpha = 1.0 / p.atr_window
    df["atr14"] = df.groupby("ticker", observed=True)["tr"].transform(
        lambda s: s.ewm(alpha=alpha, adjust=False, min_periods=p.atr_window).mean()
    )

    # Swing low for stops
    df["swing_low_L"] = by["low"].apply(lambda s: s.shift(1).rolling(p.swing_lookback, min_periods=1).min())

    return df


def forward_returns(df: pd.DataFrame, horizons: List[int] = [5, 10, 20]) -> pd.DataFrame:
    df = df.copy()
    by = df.groupby("ticker", observed=True)
    for h in horizons:
        df[f"fwd_{h}d_ret"] = by["close"].shift(-h) / df["close"] - 1.0
    return df


# ------------------------- Trade Plan & Reports ------------------------- #

def make_trade_plan(signals_only: pd.DataFrame, p: SosParams) -> pd.DataFrame:
    s = signals_only.copy()

    # Entry
    if p.entry_policy == "next_open":
        s["next_open"] = s.groupby("ticker", observed=True)["open"].shift(-1)
        s["entry_price"] = s["next_open"]
        s["entry_fallback_used"] = s["entry_price"].isna()
        s.loc[s["entry_fallback_used"], "entry_price"] = s.loc[s["entry_fallback_used"], "close"]
    elif p.entry_policy == "eod_close":
        s["entry_price"] = s["close"]
        s["entry_fallback_used"] = False
    else:
        raise ValueError(f"Unknown entry_policy: {p.entry_policy}")
    s["entry_policy"] = p.entry_policy

    # Stops
    s["stop_swing_price"] = s["swing_low_L"]
    s["stop_defense_price"] = s["low"]
    if p.stop_method == "swing_only":
        s["stop_source_price"] = s["stop_swing_price"]; s["stop_method"] = "swing_only"
    elif p.stop_method == "defense_only":
        s["stop_source_price"] = s["stop_defense_price"]; s["stop_method"] = "defense_only"
    elif p.stop_method == "min_of_both":
        s["stop_source_price"] = np.fmin(s["stop_swing_price"], s["stop_defense_price"]); s["stop_method"] = "min_of_both"
    else:
        raise ValueError(f"Unknown stop_method: {p.stop_method}")

    s["stop_buffer_atr_mult"] = p.stop_buffer_atr
    s["stop_price"] = s["stop_source_price"] - (p.stop_buffer_atr * s["atr14"])
    s["risk_per_share"] = s["entry_price"] - s["stop_price"]

    # Targets
    s["rr_multiple"] = p.rr_multiple
    s["target_2R"] = s["entry_price"] + (p.rr_multiple * s["risk_per_share"])
    s["target_3R"] = s["entry_price"] + (3.0 * s["risk_per_share"])
    s["valid_trade"] = (s["risk_per_share"] > 0) & s["stop_price"].notna() & s["entry_price"].notna()

    return s


def summarize_signals(signals: pd.DataFrame, horizons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (signals
               .groupby("ticker", observed=True)
               .agg(n_signals=("date", "count"),
                    last_signal=("date", "max"))
               .reset_index()
               .sort_values(["n_signals", "ticker"], ascending=[False, True]))

    agg_list = []
    row = {"ticker": "__ALL__"}
    for h in horizons:
        col = f"fwd_{h}d_ret"
        sub = signals[col].dropna() if col in signals.columns else pd.Series([], dtype=float)
        if len(sub) == 0:
            row.update({f"mean_{h}d": np.nan, f"median_{h}d": np.nan,
                        f"winrate_{h}d": np.nan, f"n_{h}d": 0})
        else:
            row.update({
                f"mean_{h}d": sub.mean(),
                f"median_{h}d": sub.median(),
                f"winrate_{h}d": (sub > 0).mean(),
                f"n_{h}d": int(sub.count())
            })
    agg_list.append(row)

    for tkr, g in signals.groupby("ticker", observed=True):
        row = {"ticker": tkr}
        for h in horizons:
            col = f"fwd_{h}d_ret"
            sub = g[col].dropna() if col in g.columns else pd.Series([], dtype=float)
            if len(sub) == 0:
                row.update({f"mean_{h}d": np.nan, f"median_{h}d": np.nan,
                            f"winrate_{h}d": np.nan, f"n_{h}d": 0})
            else:
                row.update({
                    f"mean_{h}d": sub.mean(),
                    f"median_{h}d": sub.median(),
                    f"winrate_{h}d": (sub > 0).mean(),
                    f"n_{h}d": int(sub.count())
                })
        agg_list.append(row)

    eval_summary = pd.DataFrame(agg_list)
    return summary, eval_summary


def to_sqlite(
    db_path: str,
    prices: Optional[pd.DataFrame] = None,
    signals: Optional[pd.DataFrame] = None,
    eval_summary: Optional[pd.DataFrame] = None
) -> None:
    con = sqlite3.connect(db_path)
    try:
        if prices is not None and not prices.empty:
            prices.to_sql("prices_clean", con, if_exists="replace", index=False)
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_prices_clean ON prices_clean(ticker, date)")

        if signals is not None and not signals.empty:
            signals.to_sql("sos_signals", con, if_exists="replace", index=False)
            con.execute("CREATE INDEX IF NOT EXISTS ix_sos_signals_ticker_date ON sos_signals(ticker, date)")
            con.execute("CREATE INDEX IF NOT EXISTS ix_sos_signals_date ON sos_signals(date)")

        if eval_summary is not None and not eval_summary.empty:
            eval_summary.to_sql("sos_eval_summary", con, if_exists="replace", index=False)
    finally:
        con.commit()
        con.close()


# ------------------------- Main ------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Show-of-Strength signal scanner (adjusted-basis aware)")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV (quote if it has spaces or &)")
    parser.add_argument("--min-rows", type=int, default=30, help="Min rows per symbol to include")

    # SoS rule params
    parser.add_argument("--prior-high-window", type=int, default=5)
    parser.add_argument("--ema-span", type=int, default=9)
    parser.add_argument("--vol-sma-window", type=int, default=20)
    parser.add_argument("--volume-mult", type=float, default=1.2)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--rsi-threshold", type=float, default=50.0)
    parser.add_argument("--pullback-window", type=int, default=10)
    parser.add_argument("--min-days-below-ema", type=int, default=2)
    parser.add_argument("--rsi-pullback-threshold", type=float, default=45.0)

    # MA50 proximity
    parser.add_argument("--ema50-span", type=int, default=50)
    parser.add_argument("--ma50-type", choices=["ema", "sma"], default="ema",
                        help="Which 50-bar average to use for proximity: EMA (default) or SMA.")
    parser.add_argument("--ema50-prox-pct", type=float, default=0.02)
    parser.add_argument("--ema50-lookback", type=int, default=10)
    parser.add_argument("--ema50-prox-mode", choices=["lookback", "today", "recent"], default="lookback")
    parser.add_argument("--ema50-recent-days", type=int, default=4)
    parser.add_argument("--disable-ema50-proximity", action="store_true",
                        help="Disable the 50MA proximity requirement")

    # Extra guards
    parser.add_argument("--min-breakout-pct", type=float, default=0.0,
                        help="Require close/prior_high - 1 >= this (e.g., 0.01 for +1%)")
    parser.add_argument("--min-range-pos", type=float, default=0.0,
                        help="Require (close-low)/(high-low) >= this (0..1). Example: 0.6")

    # Basis sanity
    parser.add_argument("--basis-window", type=int, default=60,
                        help="Rolling window for basis sanity median gap")
    parser.add_argument("--basis-gap-thresh", type=float, default=0.30,
                        help="Median |close-MA50|/close threshold to flag suspect basis")
    parser.add_argument("--exclude-suspect-basis", action="store_true", default=True,
                        help="Exclude rows flagged by basis sanity")

    # Trade plan
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.25)
    parser.add_argument("--swing-lookback", type=int, default=10)
    parser.add_argument("--stop-method", choices=["swing_only", "defense_only", "min_of_both"], default="min_of_both")
    parser.add_argument("--entry-policy", choices=["next_open", "eod_close"], default="next_open")
    parser.add_argument("--rr-multiple", type=float, default=2.0)

    # Evaluation horizons
    parser.add_argument("--eval-horizons", nargs="+", type=int, default=[5, 10, 20])

    # Output file paths
    parser.add_argument("--signals-csv", default="show_of_strength_signals.csv")
    parser.add_argument("--summary-csv", default="show_of_strength_summary.csv")
    parser.add_argument("--eval-csv", default="show_of_strength_eval.csv")

    # Recent window controls
    parser.add_argument("--recent-days", type=int, default=7,
                        help="Days back from dataset max(date) for recent CSV")
    parser.add_argument("--recent-csv", default=None,
                        help="Filename for recent signals CSV (defaults to show_of_strength_recent_<N>d.csv)")
    parser.add_argument("--recent-only", action="store_true", help="Only produce the recent signals CSV")

    parser.add_argument("--write-db", action="store_true", help="Write outputs to SQLite")
    parser.add_argument("--db-path", default="signals.db")

    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}. If your filename contains & or spaces, wrap it in quotes.")

    logging.info("Loading and cleaning CSV…")
    prices = parse_and_clean(str(csv_path), min_rows_per_symbol=args.min_rows)
    logging.info(f"Cleaned prices: {len(prices):,} rows across {prices['ticker'].nunique()} symbols "
                 f"from {prices['date'].min().date()} to {prices['date'].max().date()}")

    # Handle legacy misspelling safely
    bg_thresh = getattr(args, "basis_gap_thresh", getattr(args, "basis_gap_thres", 0.30))

    p = SosParams(
        prior_high_window=args.prior_high_window,
        ema_span=args.ema_span,
        vol_sma_window=args.vol_sma_window,
        volume_mult=args.volume_mult,
        rsi_window=args.rsi_window,
        rsi_threshold=args.rsi_threshold,
        pullback_window=args.pullback_window,
        min_days_below_ema=args.min_days_below_ema,
        rsi_pullback_threshold=args.rsi_pullback_threshold,
        ema50_span=args.ema50_span,
        ma50_type=args.ma50_type,
        ema50_prox_pct=args.ema50_prox_pct,
        ema50_lookback=args.ema50_lookback,
        ema50_prox_mode=args.ema50_prox_mode,
        ema50_recent_days=args.ema50_recent_days,
        ema50_required=not args.disable_ema50_proximity,
        min_breakout_pct=args.min_breakout_pct,
        min_range_pos=args.min_range_pos,
        basis_window=args.basis_window,
        basis_gap_thresh=bg_thresh,
        exclude_suspect_basis=args.exclude_suspect_basis,
        atr_window=args.atr_window,
        stop_buffer_atr=args.stop_buffer_atr,
        swing_lookback=args.swing_lookback,
        stop_method=args.stop_method,
        entry_policy=args.entry_policy,
        rr_multiple=args.rr_multiple
    )

    logging.info("Computing indicators and SoS conditions…")
    enriched = compute_indicators(prices, p)

    # Forward returns only when not recent-only
    if not args.recent_only:
        logging.info("Computing forward returns…")
        enriched = forward_returns(enriched, horizons=args.eval_horizons)

    # Extract signals
    signals = enriched[enriched["signal_sos"]].copy()

    # Trade plan
    logging.info("Building trade plans (entry/stop/targets)…")
    signals = make_trade_plan(signals, p)

    # Column blocks for outputs
    base_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    ind_cols = [
        "ema_fast", "prior_high_n", "vol_sma20", "rsi", "volume_ratio",
        "close_vs_ema_pct", "close_vs_prior_high_pct", "range_pos",
        "ema50", "ema50_dist_close_pct", "ema50_dist_low_pct",
        "ema50_proximity_today", "ema50_proximity_lookback",
        "ema50_proximity_recentK", "ema50_days_since_touch",
        "atr14", "swing_low_L", "ema50_gap_pct", "basis_suspect"
    ]
    cond_cols = [
        "cond_above_ema", "cond_breakout", "cond_volume",
        "cond_rsi_regime", "cond_rsi_rising", "had_pullback",
        "cond_min_breakout", "cond_min_rangepos", "cond_ema50_prox", "signal_sos"
    ]
    trade_cols = [
        "entry_policy", "entry_price", "entry_fallback_used",
        "stop_method", "stop_source_price", "stop_buffer_atr_mult", "stop_price",
        "risk_per_share", "rr_multiple", "target_2R", "target_3R", "valid_trade"
    ]

    # --- recent-only branch (anchor to dataset max date) ---
    if args.recent_only:
        anchor_date = prices["date"].max().normalize()
        min_recent = anchor_date - pd.Timedelta(days=args.recent_days - 1)
        signals_recent = (signals[signals["date"] >= min_recent]
                          .sort_values(["date", "ticker"])
                          .reset_index(drop=True))

        # Ensure all expected cols exist even if empty
        recent_cols = base_cols + ind_cols + cond_cols + trade_cols
        for c in recent_cols:
            if c not in signals_recent.columns:
                signals_recent[c] = np.nan

        out_recent = signals_recent[recent_cols].copy()
        recent_name = args.recent_csv or f"show_of_strength_recent_{args.recent_days}d.csv"
        out_recent.to_csv(recent_name, index=False)
        logging.info(f"Wrote recent signals ({args.recent_days}d) → {recent_name} "
                     f"({len(out_recent):,} rows from {min_recent.date()} to {anchor_date.date()})")

        if args.write_db:
            logging.info(f"Writing recent signals to SQLite at {args.db_path} …")
            to_sqlite(args.db_path, prices=None, signals=out_recent, eval_summary=None)
            logging.info("SQLite write complete.")
        logging.info("Done.")
        return

    # --- full outputs ---
    fwd_cols = [f"fwd_{h}d_ret" for h in args.eval_horizons]
    out_cols = base_cols + ind_cols + cond_cols + trade_cols + fwd_cols
    signals_out = signals[out_cols].sort_values(["date", "ticker"]).reset_index(drop=True)

    logging.info(f"Signals found: {len(signals_out):,}")
    signals_out.to_csv(args.signals_csv, index=False)
    logging.info(f"Wrote signals → {args.signals_csv}")

    summary, eval_summary = summarize_signals(signals_out, args.eval_horizons)
    summary.to_csv(args.summary_csv, index=False)
    eval_summary.to_csv(args.eval_csv, index=False)
    logging.info(f"Wrote per-ticker summary → {args.summary_csv}")
    logging.info(f"Wrote evaluation summary → {args.eval_csv}")

    # Also write the recent-<N>d CSV anchored to dataset max date
    anchor_date = prices["date"].max().normalize()
    min_recent = anchor_date - pd.Timedelta(days=args.recent_days - 1)
    signals_recent = (signals_out[signals_out["date"] >= min_recent]
                      .sort_values(["date", "ticker"])
                      .reset_index(drop=True))
    recent_cols = base_cols + ind_cols + cond_cols + trade_cols
    out_recent = signals_recent[recent_cols].copy()
    recent_name = args.recent_csv or f"show_of_strength_recent_{args.recent_days}d.csv"
    out_recent.to_csv(recent_name, index=False)
    logging.info(f"Wrote recent signals ({args.recent_days}d) → {recent_name} "
                 f"({len(out_recent):,} rows from {min_recent.date()} to {anchor_date.date()})")

    if args.write_db:
        logging.info(f"Writing to SQLite at {args.db_path} …")
        prices_to_db = enriched[[
            "date","ticker","open","high","low","close","volume",
            "ema_fast","rsi","vol_sma20","prior_high_n","ema50","atr14"
        ]]
        to_sqlite(args.db_path, prices=prices_to_db, signals=signals_out, eval_summary=eval_summary)
        logging.info("SQLite write complete.")

    logging.info("Done.")


if __name__ == "__main__":
    main()
