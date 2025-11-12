# Irish Day-Ahead Power Price Forecasting (I-SEM)

Production-ready starter to forecast **SEMOpx Day-Ahead Market (DAM)** prices using ENTSO-E, EirGrid proxies and weather features.
It’s designed for interview-readability and extension to XGBoost/LSTM.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ENTSOE_TOKEN
# pull data
python scripts/fetch_entsoe.py --start 2021-01-01 --end 2025-09-30
python scripts/fetch_weather.py --start 2021-01-01 --end 2025-09-30
# build + train + evaluate
python scripts/make_dataset.py
python scripts/train.py
python scripts/evaluate.py
# dashboard
streamlit run src/dashboard/app.py
```
See `config.yaml` for settings (zone, timezone, weather coords).

## Updating SEMOpx data

The app does **not** scrape SEMOpx live. Instead it uses the official
“lookback” workbooks as the source of truth for historical Irish DAM prices.

To refresh the data:

1. Download the latest **SEMOpx Ex-Ante Market look back** files:
   - `lookback_mkt.xlsx` (older history)
   - `Lookback2_mkt.xlsx` (recent history)
   from the SEMOpx Document Library.

2. Save them into the repo at:

   ```text
   src/data/raw/lookback_mkt.xlsx
   src/data/raw/Lookback2_mkt.xlsx
