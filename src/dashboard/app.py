# --- add near the top of src/dashboard/app.py (new branch, via PR) ---
import os
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/train.parquet")

@st.cache_data(show_spinner=True)
def ensure_dataset():
    if DATA_PATH.exists():
        return
    # Lazy build: call the same code the scripts use
    from src.data.entsoe_api import Entsoe
    from src.data.weather import fetch_hourly
    from src.features.build_features import build_feature_table
    from src.features.targets import make_day_ahead_target
    import yaml, os
    from dotenv import load_dotenv; load_dotenv()

    cfg = yaml.safe_load(open("config.yaml"))
    ent = Entsoe()
    # Pull a modest window so cold starts are fast
    start = cfg["train"]["start"]
    end   = cfg["train"]["end"]

    dam = ent.day_ahead_prices(start, end)
    load_fc = ent.load_forecast(start, end)
    ws = ent.wind_solar_forecast(start, end)

    lat = cfg["weather"]["lat"]; lon = cfg["weather"]["lon"]
    weather = fetch_hourly(lat, lon, start, end)

    X = build_feature_table(dam, load_fc, ws, weather)
    y = make_day_ahead_target(dam).reindex(X.index)
    df = X.copy(); df["target"] = y
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.dropna().to_parquet(DATA_PATH)

ensure_dataset()

df = pd.read_parquet(DATA_PATH)


import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Irish DAM Forecast', layout='wide')

@st.cache_data
def load():
    return pd.read_parquet('data/processed/train.parquet')


df = load()

y = df.pop('target')
mdl = __import__('src.models.xgb_model', fromlist=['make_model']).make_model()
mdl.fit(df, y)

st.title('Irish Day‑Ahead Power Price Forecast (SEMOpx)')

from datetime import datetime
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo('Europe/Dublin')).date()

date = st.date_input('Choose a date to forecast', now)
mask = (df.index.date == pd.to_datetime(date).date())
X_day = df.loc[mask]
if len(X_day):
    preds = mdl.predict(X_day)
    out = pd.DataFrame({'ts': X_day.index, 'forecast_eur_mwh': preds}).set_index('ts')
    st.subheader('Hourly forecast')
    st.dataframe(out)
    st.plotly_chart(px.line(out, y='forecast_eur_mwh'), use_container_width=True)
else:
    st.info('No features for this day yet. Run data fetch scripts for the target date range.')
