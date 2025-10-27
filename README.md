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
