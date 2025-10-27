# src/dashboard/app.py

# --- bootstrap (imports first!) ---
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on sys.path so `import src...` works in Streamlit Cloud/Codespaces
APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[2]  # .../irish-dam-forecast
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

load_dotenv()  # local dev uses .env; on Streamlit Cloud use st.secrets

DATA_PATH = Path("data/processed/train.parquet")


@st.cache_data(show_spinner=True)
def ensure_dataset():
    """Build data/processed/train.parquet once, using ENTSO-E + Open-Meteo."""
    if DATA_PATH.exists():
        return  # already built

    # Heavy imports only when needed (after sys.path fix above)
    from src.data.entsoe_api import Entsoe
    from src.data.weather import fetch_hourly
    from src.features.build_features import build_feature_table
    from src.features.targets import make_day_ahead_target

    cfg = yaml.safe_load(open("config.yaml"))

    # inside ensure_dataset(), after loading cfg
from datetime import datetime, timedelta, timezone

def _as_date(s):  # returns None if empty/invalid
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

start_cfg = _as_date(cfg["train"].get("start"))
end_cfg   = _as_date(cfg["train"].get("end"))

if not end_cfg:
    end_cfg = datetime.now(timezone.utc).date()
if not start_cfg:
    start_cfg = (pd.to_datetime(end_cfg) - pd.Timedelta(days=90)).date()

start = str(start_cfg)
end   = str(end_cfg)


    # Token priority: ENV (local) -> Streamlit secrets (cloud)
    token = os.getenv("ENTSOE_TOKEN") or st.secrets.get("ENTSOE_TOKEN")
    if not token:
        st.error(
            "ENTSOE_TOKEN not found. Set it in .env (local) or in Streamlit → Settings → Secrets."
        )
        st.stop()

    ent = Entsoe(token=token)

    start = cfg["train"]["start"]
    end = cfg["train"]["end"]

    # Pull data
    dam = ent.day_ahead_prices(start, end)
    load_fc = ent.load_forecast(start, end)
    ws = ent.wind_solar_forecast(start, end)

    lat = cfg["weather"]["lat"]
    lon = cfg["weather"]["lon"]
    weather = fetch_hourly(lat, lon, start, end)

    # Features + target
    X = build_feature_table(dam, load_fc, ws, weather)
    y = make_day_ahead_target(dam).reindex(X.index)

    out = X.copy()
    out["target"] = y

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.dropna().to_parquet(DATA_PATH)


# Build (cached) then load
ensure_dataset()
df = pd.read_parquet(DATA_PATH)

# --- app proper ---
import plotly.express as px
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Irish DAM Forecast", layout="wide")

# Train quick model on all rows
y = df.pop("target")
mdl = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
mdl.fit(df, y)

st.title("Irish Day-Ahead Power Price Forecast (SEMOpx)")

today_ie = datetime.now(ZoneInfo("Europe/Dublin")).date()
date = st.date_input("Choose a date to forecast", today_ie)

mask = (df.index.date == pd.to_datetime(date).date())
X_day = df.loc[mask]

if len(X_day):
    preds = mdl.predict(X_day)
    out = pd.DataFrame({"ts": X_day.index, "forecast_eur_mwh": preds}).set_index("ts")
    st.subheader("Hourly forecast")
    st.dataframe(out)
    st.plotly_chart(px.line(out, y="forecast_eur_mwh"), use_container_width=True)
else:
    st.info(
        "No features for this day yet. Adjust the training window in config.yaml "
        "or let the app build more data on first run."
    )
