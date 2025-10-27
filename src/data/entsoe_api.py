# src/data/entsoe_api.py
from __future__ import annotations
import os
import pandas as pd
from entsoe import EntsoePandasClient
from zoneinfo import ZoneInfo

IE = "IE"  # Ireland bidding zone


class Entsoe:
    def __init__(self, token: str | None = None, tz: str = "Europe/Dublin"):
        token = token or os.getenv("ENTSOE_TOKEN")
        if not token:
            raise RuntimeError("ENTSOE_TOKEN not set. Put it in .env or st.secrets.")
        self.client = EntsoePandasClient(api_key=token)
        self.tz = ZoneInfo(tz)

    def _ts(self, dt_like) -> pd.Timestamp:
        # Normalize to tz-aware Dublin timestamps
        return pd.Timestamp(dt_like, tz=self.tz)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        s = self.client.query_day_ahead_prices(
            IE,
            start=self._ts(start),
            end=self._ts(end),
        )
        s.name = "dam_eur_mwh"
        return s

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        df = self.client.query_wind_solar_forecast(
            IE,
            start=self._ts(start),
            end=self._ts(end),
        )
        # Flatten MultiIndex columns like ('DayAheadGenerationForecast', 'Solar')
        df.columns = [f"{a}_{b}".lower().replace(" ", "_") for a, b in df.columns]
        return df

    def load_forecast(self, start: str, end: str) -> pd.Series:
        s = self.client.query_load_forecast(
            IE,
            start=self._ts(start),
            end=self._ts(end),
        )
        s.name = "load_forecast_mw"
        return s
