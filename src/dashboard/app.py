# src/dashboard/app.py

# --- bootstrap (imports first!) ---
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import yaml
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on sys.path so `import src...` works in Streamlit Cloud/Codespaces
APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[2]  # .../irish-dam-forecast
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

load_dotenv()  # local dev uses .env; on Streamlit Cloud we'll use st.secrets

DATA_PATH = Path("data/processed/train.parquet")


def _as_date(s):
    """Parse s to date; return None if missing/blank/invalid."""
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _chunk_edges(start_date, end_date, days=120):
    """Yield [start,end] chunks inclusive with <= days span (ENTSO-E safe)."""
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    step = pd.Timedelta(days=days)
    cur = pd.to_datetime(s)
    last = pd.to_datetime(e)
    while cur <= last:
        nxt = min(cur + step, last)
        yield str(cur.date()), str(nxt.date())
        cur = nxt + pd.Timedelta(days=1)


@st.cache_data(show_spinner=True)
def ensure_dataset():
    """Build data/processed/train.parquet once, using ENTSO-E + Open-Meteo, chunked and tolerant to empty chunks."""
    if DATA_PATH.exists():
        return  # already built

    # Heavy imports only when needed (after sys.path fix above)
    from src.data.entsoe_api import Entsoe
    from src.data.weather import fetch_hourly
    from src.features.build_features import build_feature_table
    from src.features.targets import make_day_ahead_target
    # import here to avoid hard dependency if entsoe-py changes
    from entsoe.exceptions import NoMatchingDataError

    # Load config and compute a safe window if missing
    cfg = yaml.safe_load(open("config.yaml"))
    start_cfg = _as_date(cfg.get("train", {}).get("start"))
    end_cfg = _as_date(cfg.get("train", {}).get("end"))
    if not end_cfg:
        end_cfg = datetime.now(timezone.utc).date()
    if not start_cfg:
        start_cfg = (pd.to_datetime(end_cfg) - pd.Timedelta(days=90)).date()
    start_all = str(start_cfg)
    end_all = str(end_cfg)

    # Token: ENV first, then Streamlit secrets
    token = os.getenv("ENTSOE_TOKEN") or st.secrets.get("ENTSOE_TOKEN")
    if not token:
        st.error("ENTSOE_TOKEN not found. Set it in .env (local) or in Streamlit → Settings → Secrets.")
        st.stop()

    ent = Entsoe(token=token)

    # ---- Chunked pulls to avoid 400 and tolerate empty chunks ----
    def _pull_series(method_name):
        parts = []
        empty_spans = []
        for s, e in _chunk_edges(start_all, end_all, days=120):
            for attempt in range(3):
                try:
                    srs = getattr(ent, method_name)(start=s, end=e)
                    if srs is not None and len(srs) > 0:
                        parts.append(srs)
                    else:
                        empty_spans.append((s, e))
                    break
                except NoMatchingDataError:
                    empty_spans.append((s, e))
                    break
                except Exception:
                    # simple backoff on 400/429/5xx
                    time.sleep(1.5 * (attempt + 1))
                    if attempt == 2:
                        raise
        if empty_spans and not parts:
            st.warning(f"ENTSO-E returned no data for {method_name} in window {start_all} → {end_all}. "
                       f"Try narrowing the window in config.yaml (e.g., last 90–180 days).")
        if not parts:
            return pd.Series(dtype=float)
        srs = pd.concat(parts).sort_index()
        srs = srs[~srs.index.duplicated(keep="last")]
        return srs

    def _pull_frame(method_name):
        parts = []
        empty_spans = []
        for s, e in _chunk_edges(start_all, end_all, days=120):
            for attempt in range(3):
                try:
                    dfp = getattr(ent, method_name)(start=s, end=e)
                    if dfp is not None and not dfp.empty:
                        parts.append(dfp)
                    else:
                        empty_spans.append((s, e))
                    break
                except NoMatchingDataError:
                    empty_spans.append((s, e))
                    break
                except Exception:
                    time.sleep(1.5 * (attempt + 1))
                    if attempt == 2:
                        raise
        if empty_spans and not parts:
            st.warning(f"ENTSO-E returned no data for {method_name} in window {start_all} → {end_all}.")
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    st.caption(f"Fetching ENTSO-E (chunked): {start_all} → {end_all}")
    dam = _pull_series("day_ahead_prices")
    load_fc = _pull_series("load_forecast")
    ws = _pull_frame("wind_solar_forecast")

    # Hard guards: we need at least prices and load forecast to proceed
    if dam.empty:
        st.error("No day-ahead prices returned for the selected window. "
                 "Set a recent window in config.yaml (e.g., last 90–180 days) and rerun.")
        st.stop()
    if load_fc.empty:
        st.error("No load forecast returned for the selected window. "
                 "Try a recent window in config.yaml and rerun.")
        st.stop()

    # Weather in one go (Open-Meteo tolerates larger ranges)
    lat = cfg["weather"]["lat"]
    lon = cfg["weather"]["lon"]
    weather = fetch_hourly(lat, lon, start_all, end_all)

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
from datetime import datetime as dtmod
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Irish DAM Forecast", layout="wide")

# Train quick model on all rows
y = df.pop("target")
mdl = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
mdl.fit(df, y)

st.title("Irish Day-Ahead Power Price Forecast (SEMOpx)")

today_ie = dtmod.now(ZoneInfo("Europe/Dublin")).date()
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
