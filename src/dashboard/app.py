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


def _chunk_edges(start_date, end_date, days=30):  # 30 keeps calls small/fast
    s = pd.to_datetime(start_date).date()
    e = pd.to_datetime(end_date).date()
    step = pd.Timedelta(days=days)
    cur = pd.to_datetime(s); last = pd.to_datetime(e)
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
    from entsoe.exceptions import NoMatchingDataError

    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Compute window (defaults if missing)
    start_cfg = _as_date(cfg.get("train", {}).get("start"))
    end_cfg   = _as_date(cfg.get("train", {}).get("end"))
    if not end_cfg:
        end_cfg = datetime.now(timezone.utc).date()
    if not start_cfg:
        start_cfg = (pd.to_datetime(end_cfg) - pd.Timedelta(days=90)).date()
    start_all = str(start_cfg)
    end_all   = str(end_cfg)

    # --- FAST MODE: clamp window to last N days so first build is quick ---
    MAX_DAYS = int(os.getenv("MAX_DAYS", "60"))  # tune 30–90 as you like
    span_days = (pd.to_datetime(end_all) - pd.to_datetime(start_all)).days + 1
    if span_days > MAX_DAYS:
        start_all = str((pd.to_datetime(end_all) - pd.Timedelta(days=MAX_DAYS)).date())
        st.info(f"Fast mode: clamped window to last {MAX_DAYS} days → {start_all} → {end_all}")

    # Token: ENV first, then Streamlit secrets
    token = os.getenv("ENTSOE_TOKEN") or st.secrets.get("ENTSOE_TOKEN")
    if not token:
        st.error("ENTSOE_TOKEN not found. Set it in .env (local) or in Streamlit → Settings → Secrets.")
        st.stop()

    ent = Entsoe(token=token)

    # ---- Chunked pulls to avoid 400s and tolerate empty chunks ----
    def _pull_series(method_name: str) -> pd.Series:
        parts, empty_spans = [], []
        for i, (s, e) in enumerate(_chunk_edges(start_all, end_all, days=30), 1):
            # Optional: comment out next line if you don’t want logs on the page
            st.write(f"ENTSO-E {method_name} chunk {i}: {s} → {e}")
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
                    time.sleep(1.5 * (attempt + 1))
                    if attempt == 2:
                        raise
        if not parts:
            return pd.Series(dtype=float)
        srs = pd.concat(parts).sort_index()
        srs = srs[~srs.index.duplicated(keep="last")]
        return srs

    def _pull_frame(method_name: str) -> pd.DataFrame:
        parts, empty_spans = [], []
        for i, (s, e) in enumerate(_chunk_edges(start_all, end_all, days=30), 1):
            # Optional: comment out next line if you don’t want logs on the page
            st.write(f"ENTSO-E {method_name} chunk {i}: {s} → {e}")
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
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    st.caption(f"Fetching ENTSO-E (chunked): {start_all} → {end_all}")
    dam     = _pull_series("day_ahead_prices")
    load_fc = _pull_series("load_forecast")
    ws      = _pull_frame("wind_solar_forecast")

    # Hard guards
    if dam.empty:
        st.error("No day-ahead prices returned. Set a recent window in config.yaml (e.g., last 90–180 days) and rerun.")
        st.stop()
    if load_fc.empty:
        st.error("No load forecast returned. Try a recent window in config.yaml and rerun.")
        st.stop()

    # Weather
    lat = cfg["weather"]["lat"]; lon = cfg["weather"]["lon"]
    weather = fetch_hourly(lat, lon, start_all, end_all)

    # ---- Timezone harmonization (make all indices Europe/Dublin tz-naive) ----
    def _to_local_naive(obj, assume_utc: bool = False):
        if not hasattr(obj, "index") or not isinstance(obj.index, pd.DatetimeIndex):
            return obj
        out = obj.copy()
        idx = out.index
        if idx.tz is None:
            base_tz = "UTC" if assume_utc else "Europe/Brussels"
            idx = idx.tz_localize(base_tz)
        idx = idx.tz_convert("Europe/Dublin").tz_localize(None)
        out.index = idx
        return out

    dam     = _to_local_naive(dam, assume_utc=False)
    load_fc = _to_local_naive(load_fc, assume_utc=False)
    ws      = _to_local_naive(ws, assume_utc=False)
    weather = _to_local_naive(weather, assume_utc=True)

    # ---- Ensure load_fc is a Series named 'load_forecast_mw' ----
    if isinstance(load_fc, pd.DataFrame):
        def _norm(c): return " ".join(map(str, c)).lower() if isinstance(c, tuple) else str(c).lower()
        colmap = {_norm(c): c for c in load_fc.columns}
        pick = None
        for key in ("load forecast", "forecast", "load"):
            for k, c in colmap.items():
                if key in k:
                    pick = c; break
            if pick is not None:
                break
        if pick is None:
            pick = load_fc.columns[0]
        load_fc = load_fc[pick]
    load_fc = pd.to_numeric(load_fc, errors="coerce").astype("float64")
    load_fc.name = "load_forecast_mw"

    # ---- Build features + target ----
    X = build_feature_table(dam, load_fc, ws, weather)
    y = make_day_ahead_target(dam).reindex(X.index)

    if X.empty:
        st.error(
            "No overlapping timestamps across ENTSO-E & weather feeds.\n"
            "Try narrowing the window in config.yaml (e.g., 2025-07-01 → 2025-09-30), "
            "or verify the feeds returned data."
        )
        st.stop()

    out = X.copy()
    out["target"] = y
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.dropna().to_parquet(DATA_PATH)


# Build (cached) then load
ensure_dataset()
df = pd.read_parquet(DATA_PATH)

# --- app proper (keep a single date picker) ---
import plotly.express as px
from datetime import date as _date
from zoneinfo import ZoneInfo

# compute coverage from the DataFrame index
idx = pd.DatetimeIndex(df.index)
days = pd.Index(idx.date)
if len(days) == 0:
    st.error("No rows found in the dataset.")
    st.stop()

def _as_pydate(d):
    return d if isinstance(d, _date) else pd.to_datetime(d).date()

day_min = _as_pydate(days.min())
day_max = _as_pydate(days.max())
today_ie = datetime.now(ZoneInfo("Europe/Dublin")).date()
default_day = max(day_min, min(day_max, today_ie))

st.caption(f"Data coverage: {day_min} → {day_max} | rows: {len(df):,}")

date = st.date_input(
    "Choose a date to forecast",
    value=default_day,
    min_value=day_min,
    max_value=day_max,
)

# Train quick model on all rows
y = df.pop("target")
mdl = __import__("src.models.xgb_model", fromlist=["make_model"]).make_model()
mdl.fit(df, y)

st.title("Irish Day-Ahead Power Price Forecast (SEMOpx)")

X_day = df.loc[df.index.date == pd.to_datetime(date).date()]
if len(X_day):
    preds = mdl.predict(X_day)
    out = pd.DataFrame({"ts": X_day.index, "forecast_eur_mwh": preds}).set_index("ts")
    st.subheader("Hourly forecast")
    st.dataframe(out)
    st.plotly_chart(px.line(out, y="forecast_eur_mwh"), use_container_width=True)
else:
    st.info("No features for this day yet. Adjust the training window in config.yaml or fetch more data.")
