# Irish Day-Ahead Power Price Forecasting (I-SEM)

Production-ready starter to forecast **SEMOpx Day-Ahead Market (DAM)** prices from local HRP60 files.
It’s designed for interview-readability and extension to XGBoost/LSTM.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# fetch SEMOpx HRP60 to local file
python scripts/fetch_semopx_dam_60min_hrp.py --start 2024-01-01 --end 2026-01-01

# build dataset
python scripts/make_dataset.py --days 365 --horizon 24

# run dashboard
streamlit run src/dashboard/app.py
```
