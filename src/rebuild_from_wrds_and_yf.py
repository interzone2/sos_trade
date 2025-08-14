#!/usr/bin/env python3
"""
Rebuild an adjusted-basis master OHLCV CSV from a WRDS export + Yahoo patch.

Input (WRDS): date,ticker,open,high,low,close,adjclose,volume
Output:       date,ticker,open,high,low,close,adjclose,volume   (ALL ADJUSTED)

- Cleans & de-duplicates WRDS rows
- Rebases O/H/L/C to adjusted basis using adjclose/close (volume untouched)
- Fetches Yahoo ADJUSTED bars (auto_adjust=True) for the chosen date window
- Merges, de-dupes on (ticker,date), “patch wins” on overlaps
- Writes a patch report and a basis sanity report

Usage examples:
  python rebuild_from_wrds_and_yf.py --wrds-csv wrds_new.csv --out-csv sp_500_new.csv
  python rebuild_from_wrds_and_yf.py --wrds-csv wrds_new.csv --out-csv sp_500_new.csv --end 2025-08-14
  python rebuild_from_wrds_and_yf.py --wrds-csv wrds_new.csv --tickers sales_growth.txt --out-csv sp_500_new.csv
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import List
import pandas as pd

def read_tickers_file(path: str) -> List[str]:
    s = Path(path).read_text(encoding="utf-8")
    toks = [t.strip().upper() for t in s.replace(",", "\n").splitlines() if t.strip()]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def load_wrds_clean_rebased(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date","ticker","open","high","low","close","adjclose","volume"}
    miss = required - set(df.columns)
    if miss:
        raise SystemExit(f"WRDS CSV missing columns: {miss}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ["open","high","low","close","adjclose","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper()

    # basic hygiene
    df = df.dropna(subset=["date","ticker","open","high","low","close","adjclose","volume"])
    df = df[(df["open"]>0)&(df["high"]>0)&(df["low"]>0)&(df["close"]>0)&(df["volume"]>0)&(df["high"]>=df["low"])]

    # rebase to adjusted basis (idempotent if adjclose==close)
    df = df.sort_values(["ticker","date"], kind="mergesort")
    factor = (df["adjclose"] / df["close"]).replace([pd.NA, float("inf")], pd.NA)
    # forward/back-fill factor per ticker to bridge occasional missing adjclose rows
    factor = (factor.groupby(df["ticker"]).transform(lambda s: s.ffill().bfill())).fillna(1.0)
    for c in ["open","high","low","close"]:
        df[c] = df[c] * factor
    df["adjclose"] = df["close"]  # clarity on adjusted basis

    # tidy + dedupe
    df = (df.sort_values(["ticker","date"])
            .drop_duplicates(["ticker","date"], keep="last")
            .reset_index(drop=True))
    return df

def fetch_yf_adjusted(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf
    # Yahoo mapping: BRK.B -> BRK-B, etc.
    ymap = {t: t.replace(".", "-") for t in tickers}
    codes = list(ymap.values())

    data = yf.download(
        tickers=" ".join(codes),
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),  # yf end exclusive
        interval="1d",
        auto_adjust=True,   # ADJUSTED O/H/L/C
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        for code in data.columns.levels[0]:
            if (code,) not in data.columns:  # skip missing
                continue
            sub = data[code].dropna(how="all")
            if sub.empty:
                continue
            ticker = next((k for k,v in ymap.items() if v==code), code)
            for idx, r in sub.iterrows():
                o,h,l,c,v = r.get("Open"),r.get("High"),r.get("Low"),r.get("Close"),r.get("Volume")
                if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c) or pd.isna(v): 
                    continue
                rows.append({
                    "date": pd.Timestamp(idx).tz_localize(None),
                    "ticker": ticker,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "adjclose": float(c),   # adjusted
                    "volume": float(v),
                })
    else:
        # single-ticker fallthrough
        sub = data.dropna(how="all")
        if not sub.empty:
            tkr = tickers[0]
            for idx, r in sub.iterrows():
                rows.append({
                    "date": pd.Timestamp(idx).tz_localize(None),
                    "ticker": tkr,
                    "open": float(r["Open"]),
                    "high": float(r["High"]),
                    "low": float(r["Low"]),
                    "close": float(r["Close"]),
                    "adjclose": float(r["Close"]),
                    "volume": float(r["Volume"]),
                })

    patch = pd.DataFrame(rows)
    if patch.empty:
        return patch

    # validate
    for c in ["open","high","low","close","adjclose","volume"]:
        patch[c] = pd.to_numeric(patch[c], errors="coerce")
    patch = patch.dropna(subset=["date","ticker","open","high","low","close","adjclose","volume"])
    patch = patch[(patch["open"]>0)&(patch["high"]>0)&(patch["low"]>0)&(patch["close"]>0)&(patch["volume"]>0)&(patch["high"]>=patch["low"])]
    patch["ticker"] = patch["ticker"].astype(str).str.upper()
    patch = (patch.sort_values(["ticker","date"])
                   .drop_duplicates(["ticker","date"], keep="last"))
    return patch

def basis_sanity(df: pd.DataFrame, window:int=60, thresh:float=0.30) -> pd.DataFrame:
    g = df.sort_values(["ticker","date"]).copy()
    # EMA50 on adjusted close
    g["ema50"] = g.groupby("ticker")["close"].transform(lambda s: s.ewm(span=50, adjust=False, min_periods=50).mean())
    tail = g.groupby("ticker").tail(window).copy()
    tail["gap"] = (tail["close"] - tail["ema50"]).abs() / tail["close"]
    stats = (tail.groupby("ticker")["gap"].median().reset_index().rename(columns={"gap":"median_gap_pct"}))
    return stats[stats["median_gap_pct"] > thresh].sort_values("median_gap_pct", ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrds-csv", required=True, help="WRDS file: date,ticker,open,high,low,close,adjclose,volume")
    ap.add_argument("--out-csv",  required=True, help="Output master CSV (adjusted basis)")
    ap.add_argument("--tickers",  help="Optional tickers file (newline/comma separated). Defaults to tickers found in WRDS.")
    ap.add_argument("--start",    help="Patch start date (YYYY-MM-DD). Defaults to max(WRDS date)+1 business day.")
    ap.add_argument("--end",      help="Patch end date (YYYY-MM-DD). Defaults to today.")
    ap.add_argument("--report-prefix", default="rebuild", help="Prefix for report CSVs")
    args = ap.parse_args()

    wrds = load_wrds_clean_rebased(args.wrds_csv)
    print(f"[WRDS] {len(wrds):,} rows | {wrds['ticker'].nunique()} tickers | {wrds['date'].min().date()} → {wrds['date'].max().date()} (adjusted basis)")

    # Universe
    if args.tickers:
        tickers = read_tickers_file(args.tickers)
        print(f"[UNIVERSE] {len(tickers)} tickers from {args.tickers}")
        wrds = wrds[wrds["ticker"].isin(tickers)]
    else:
        tickers = sorted(wrds["ticker"].unique().tolist())
        print(f"[UNIVERSE] {len(tickers)} tickers inferred from WRDS")

    # Dates
    wrds_last = wrds["date"].max().date() if not wrds.empty else None
    if args.start:
        start = dt.date.fromisoformat(args.start)
    else:
        # next business day after WRDS last (loose; Yahoo ignores non-trading days anyway)
        start = (wrds_last + dt.timedelta(days=1)) if wrds_last else dt.date.today()
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    print(f"[PATCH RANGE] {start} → {end}")

    patch = pd.DataFrame()
    if start <= end and tickers:
        patch = fetch_yf_adjusted(tickers, start, end)
        print(f"[YF] fetched {len(patch):,} rows across {patch['ticker'].nunique() if not patch.empty else 0} tickers")

    # Merge
    combined = wrds if patch.empty else pd.concat([wrds, patch], ignore_index=True)
    combined = (combined.sort_values(["ticker","date"])
                        .drop_duplicates(["ticker","date"], keep="last"))
    combined.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv} → {len(combined):,} rows | "
          f"{combined['ticker'].nunique()} tickers | "
          f"{combined['date'].min().date()} → {combined['date'].max().date()}")

    # Reports
    rep = (patch.groupby("ticker")["date"]
                 .agg(first="min", last="max", rows="count")
                 .reset_index()
                 .sort_values("ticker")) if not patch.empty else pd.DataFrame(columns=["ticker","first","last","rows"])
    rep_path = f"{args.report_prefix}_patch_report.csv"
    rep.to_csv(rep_path, index=False)
    print(f"[REPORT] wrote {rep_path}")

    suspects = basis_sanity(combined, window=60, thresh=0.30)
    sus_path = f"{args.report_prefix}_suspect_basis.csv"
    suspects.to_csv(sus_path, index=False)
    print(f"[REPORT] wrote {sus_path} (rows={len(suspects)})")

if __name__ == "__main__":
    main()
