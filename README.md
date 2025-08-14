# Show of Strength (SoS)

Tools to build an adjusted-basis OHLCV dataset (WRDS + Yahoo patch), scan for "Show of Strength" (SoS) signals, and review results in a small dashboard.

## Repo layout

```
.
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ src/
│  ├─ sos_signals.py
│  ├─ rebuild_from_wrds_and_yf.py
│  └─ sos_dashboard.py
├─ scripts/
│  └─ init_repo.sh
└─ data/
   ├─ .gitkeep
   └─ (your CSVs go here, ignored by git)
```

## Quickstart

```bash
# create venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# rebuild master from WRDS + Yahoo patch
python src/rebuild_from_wrds_and_yf.py --wrds-csv data/wrds_new.csv --out-csv data/sp_500_new.csv

# run a strict same-day proximity scan (recent only)
python src/sos_signals.py --csv 'data/sp_500_new.csv' --recent-only   --ema50-prox-mode today --ema50-prox-pct 0.015

# dashboard
streamlit run src/sos_dashboard.py
```

See `instructions.md` for many more runlines and automation.