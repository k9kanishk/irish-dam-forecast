from __future__ import annotations
import os
import pandas as pd
from entsoe import EntsoePandasClient
from zoneinfo import ZoneInfo

IE = 'IE'

class Entsoe:
    def __init__(self, token: str | None = None, tz: str = 'Europe/Dublin'):
        token = token or os.getenv('ENTSOE_TOKEN')
        if not token:
            raise RuntimeError('Set ENTSOE_TOKEN in your environment.')
        self.client = EntsoePandasClient(api_key=token)
        self.tz = ZoneInfo(tz)

    def day_ahead_prices(self, start: str, end: str) -> pd.Series:
        s = self.client.query_day_ahead_prices(IE, pd.Timestamp(start, tz=self.tz), pd.Timestamp(end, tz=self.tz))
        s.name = 'dam_eur_mwh'
        return s

    def wind_solar_forecast(self, start: str, end: str) -> pd.DataFrame:
        df = self.client.query_wind_solar_forecast(IE, pd.Timestamp(start, tz=self.tz), pd.Timestamp(end, tz=self.tz))
        df.columns = [f"{a}_{b}".lower().replace(' ', '_') for a,b in df.columns]
        return df

    def load_forecast(self, start: str, end: str) -> pd.Series:
        s = self.client.query_load_forecast(IE, pd.Timestamp(start, tz=self.tz), pd.Timestamp(end, tz=self.tz))
        s.name = 'load_forecast_mw'
        return s
