
# Show of Strength — Setup & Run Guide

This document explains how to: (1) **rebuild a clean, adjusted-basis OHLCV master** from a new **WRDS** export and **patch** the missing days from **Yahoo Finance**, (2) **run the SoS signal scan** with various rule presets (EMA/SMA proximity, volume, breakout, RSI, etc.), and (3) **review/export results** in a lightweight dashboard and/or SQLite DB.

> Assumptions: Python 3.10, macOS or Linux shell, CSV has columns: `date, ticker, open, high, low, close, adjclose, volume` (lower/upper case is fine — the scripts normalize).

---

## 0) One-time environment setup

```bash
# (recommended) create & activate a venv
python3.10 -m venv .venv
source .venv/bin/activate

# core deps
pip install --upgrade pandas numpy yfinance streamlit
```

Files used in this guide:

- `rebuild_from_wrds_and_yf.py` — builds the **master adjusted-basis CSV** by cleaning your WRDS export and patching missing days from Yahoo.
- `sos_signals.py` — runs the **Show-of-Strength** scans and produces signals + trade plans.
- `sos_dashboard.py` — Streamlit dashboard to review signals and export filtered views.
- *(optional)* `sales_growth.txt` — newline list of tickers to restrict the universe.

> Put all files in the same working directory for simplicity (or pass absolute paths).

---

## 1) Rebuild a master OHLCV from WRDS + Yahoo patch

The script **rebases O/H/L/C to adjusted basis** using `adjclose/close`, dedupes, and then **patches from the last WRDS date + 1 through today** with **Yahoo Finance adjusted** OHLCV (via `yfinance`, `auto_adjust=True`).

### 1.1 Basic rebuild (auto-patch from WRDS max date → today)

```bash
python rebuild_from_wrds_and_yf.py   --wrds-csv 'wrds_new.csv'   --out-csv 'sp_500_new.csv'
```

- Writes reports:
  - `rebuild_patch_report.csv` — first/last patched date per ticker, row counts.
  - `rebuild_suspect_basis.csv` — tickers whose median |close−MA50|/close is large (basis issues).

### 1.2 Explicit patch window (e.g., Aug 7 → 2025-08-14)

```bash
python rebuild_from_wrds_and_yf.py   --wrds-csv 'wrds_new.csv'   --out-csv 'sp_500_new.csv'   --start 2025-08-07 --end 2025-08-14
```

### 1.3 Restrict to a custom universe file

```bash
python rebuild_from_wrds_and_yf.py   --wrds-csv 'wrds_new.csv'   --tickers sales_growth.txt   --out-csv 'sp_500_new_strength.csv'
```

### 1.4 Quick sanity checks

```bash
# What is the dataset max date?
python - <<'PY'
import pandas as pd; d=pd.read_csv('sp_500_new.csv', parse_dates=['date'])
print('DATASET MAX DATE:', d['date'].max().date(), '| rows:', len(d), '| tickers:', d['ticker'].nunique())
PY

# Show patch report summary
csvcut -n rebuild_patch_report.csv 2>/dev/null || true
head -20 rebuild_patch_report.csv 2>/dev/null || true

# Show any basis suspects (should usually be few or empty)
head -20 rebuild_suspect_basis.csv 2>/dev/null || true
```

> **Adjusted-basis note:** We rebase O/H/L/C to adjusted levels when `adjclose` is present so moving averages, RSI, and proximity rules align with splits/dividends. Volumes remain raw.

---

## 2) Run the SoS signal scan (EMA/SMA proximity, volume, breakout, RSI)

`sos_signals.py` implements the full rule set, including **MA50 proximity modes** (`today`, `recent`, `lookback`), **RSI regime + rising**, **volume expansion vs 20-day average**, **breakout over prior N-day high**, and optional guards (`min_breakout_pct`, `min_range_pos`). It also builds **trade plans** (entry/stop/targets).

> **New options:** `--ma50-type {ema,sma}` to choose the 50-bar average; **recent-only** window is anchored to the **dataset’s max date**.

### 2.1 Baseline — “tagged MA50 within prior 4 sessions (not today), 1.5% band”

```bash
python sos_signals.py --csv 'sp_500_new.csv' --recent-only   --ema50-prox-mode lookback --ema50-lookback 4 --ema50-prox-pct 0.015
```

### 2.2 Same logic using `recent` (equivalent here to 4 sessions, clearer audit cols)

```bash
python sos_signals.py --csv 'sp_500_new.csv' --recent-only   --ema50-prox-mode recent --ema50-recent-days 4 --ema50-prox-pct 0.015
```

### 2.3 **Same-day** proximity to MA50 (tight bounce day)

```bash
python sos_signals.py --csv 'sp_500_new_strength.csv' --recent-only   --ema50-prox-mode today --ema50-prox-pct 0.015
```

### 2.4 Flip MA50 from EMA to **SMA** (to match IB charts more closely)

```bash
python sos_signals.py --csv 'sp_500_new_strength.csv' --recent-only   --ma50-type sma --ema50-prox-mode today --ema50-prox-pct 0.015
```

### 2.5 Stricter tape on the signal bar (volume, breakout, finish)

```bash
python sos_signals.py --csv 'sp_500_new_strength.csv' --recent-only   --ema50-prox-mode today --ema50-prox-pct 0.01   --prior-high-window 10 --min-breakout-pct 0.01   --volume-mult 1.5 --min-range-pos 0.6
```

### 2.6 Risk-first trade plan (swing-only stops, deeper R:R)

```bash
python sos_signals.py --csv 'sp_500_new.csv' --recent-only   --ema50-prox-mode lookback --ema50-lookback 4 --ema50-prox-pct 0.015   --stop-method swing_only --swing-lookback 15 --stop-buffer-atr 0.20 --rr-multiple 2.5
```

### 2.7 Be extra hard on basis consistency (filter out noisy series)

```bash
python sos_signals.py --csv 'sp_500_new.csv' --recent-only   --ema50-prox-mode recent --ema50-recent-days 4 --ema50-prox-pct 0.015   --basis-gap-thresh 0.20 --exclude-suspect-basis
```

### 2.8 Full history (not just recent), plus SQLite mirror

```bash
python sos_signals.py --csv 'sp_500_new.csv'   --ema50-prox-mode recent --ema50-recent-days 4 --ema50-prox-pct 0.015   --min-breakout-pct 0.01 --min-range-pos 0.6   --signals-csv show_of_strength_signals.csv   --summary-csv show_of_strength_summary.csv   --eval-csv show_of_strength_eval.csv   --write-db --db-path signals.db
```

> **Output files:**  
> - `show_of_strength_recent_7d.csv` (or `--recent-csv`), anchored to dataset max date.  
> - `show_of_strength_signals.csv`, `show_of_strength_summary.csv`, `show_of_strength_eval.csv` for the full run.  
> - Optional: `signals.db` SQLite with tables `prices_clean`, `sos_signals`, `sos_eval_summary`.

---

## 3) Review results in the dashboard

```bash
# install if needed
pip install --upgrade streamlit pandas numpy

# launch
streamlit run sos_dashboard.py
```

- In the sidebar, set the **CSV path** (e.g., `show_of_strength_recent_7d.csv` or `show_of_strength_signals.csv`) or upload the file.
- Filter by date range, tickers, min R:R, and **valid trades only**.
- Optional **position sizing**: input a fixed risk-per-trade (USD) to compute shares & notional.
- **Export** the filtered table with one click.

---

## 4) Useful one-liners (audit & QA)

```bash
# Dataset anchor used for --recent-only window
python - <<'PY'
import pandas as pd; d=pd.read_csv('sp_500_new_strength.csv', parse_dates=['date'])
print('DATASET MAX DATE:', d['date'].max().date())
PY

# Count signals actually in the last N dataset-days
python - <<'PY'
import pandas as pd; s=pd.read_csv('show_of_strength_recent_7d.csv', parse_dates=['date'])
print(s['date'].min(), '→', s['date'].max(), '| rows:', len(s))
PY

# Compare EMA50 vs SMA50 for a single name (e.g., CART) around the signal date
python - <<'PY'
import pandas as pd
d = pd.read_csv('sp_500_new_strength.csv', parse_dates=['date'])
g = d[d['ticker'].str.upper()=='CART'].sort_values('date').copy()
ema50 = g['close'].ewm(span=50, adjust=False, min_periods=50).mean()
sma50 = g['close'].rolling(50, min_periods=50).mean()
w = g.assign(ema50=ema50, sma50=sma50).tail(12)
print(w[['date','close','ema50','sma50']].to_string(index=False))
PY
```

---

## 5) Nightly automation (optional)

Example `cron` entry to **rebuild**, **scan**, and **sync** a shortlist nightly. This assumes your project lives at `~/new_Alden_trade/focus50_app_src_v03` and you want NYC 8:05pm.

```bash
# edit crontab
crontab -e

# add (convert 20:05 America/New_York to your server's local time)
5 20 * * 1-5 cd $HOME/new_Alden_trade/focus50_app_src_v03 && \
  source .venv/bin/activate && \
  python rebuild_from_wrds_and_yf.py --wrds-csv 'wrds_new.csv' --out-csv 'sp_500_new_strength.csv' && \
  python sos_signals.py --csv 'sp_500_new_strength.csv' --recent-only \
    --ma50-type sma --ema50-prox-mode today --ema50-prox-pct 0.01 \
    --prior-high-window 10 --min-breakout-pct 0.01 --volume-mult 1.5 --min-range-pos 0.6 \
    --signals-csv show_of_strength_signals.csv --recent-csv show_of_strength_recent_7d.csv \
    --write-db --db-path signals.db
```

---

## 6) Troubleshooting cheatsheet

- **File not found**: quote paths with spaces or `&` → `'sp_500_new.csv'`.
- **“Expecting value: line 1 column 1 (char 0)”** during Yahoo fetch: transient network or symbol mapping; retry or trim universe; remember `BRK.B → BRK-B`, etc. (the rebuild script handles `.`→`-` mapping for Yahoo).
- **Recent-only shows old dates**: fixed — window is now anchored to the **dataset max date**.
- **EMA vs SMA mismatch with IB charts**: use `--ma50-type sma` or tighten `--ema50-prox-pct` to 1.0%/0.75%, or require exact tag (we can add a `--ma50-touch-exact` flag if desired).
- **Pandas deprecation warnings**: the shipped code avoids deprecated `groupby.apply` on frames.
- **Empty shortlist**: means no names satisfied all gates in the last N dataset days; widen `--recent-days` or relax filters.

---

## 7) Quick reference — key flags

- **MA choice**: `--ma50-type {ema,sma}` (default: ema)
- **Proximity**: `--ema50-prox-mode {today,recent,lookback}`, size via `--ema50-prox-pct 0.015`
- **Breakout window**: `--prior-high-window 5` (try 10 for swing highs)
- **Tape quality**: `--volume-mult 1.5`, `--min-breakout-pct 0.01`, `--min-range-pos 0.6`
- **RSI regime**: `--rsi-threshold 50` (use 55 for stronger regimes)
- **Stops/targets**: `--stop-method {swing_only,defense_only,min_of_both}`, `--stop-buffer-atr 0.25`, `--rr-multiple 2.0`
- **Recent window**: `--recent-only --recent-days 7` (anchored to dataset max date)
- **DB export**: `--write-db --db-path signals.db`

---

*Updated: 2025-08-14*
