from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.base import DataQualityReport, DataSource
from src.data.weather import fetch_hourly


class WeatherSource(DataSource):
    def __init__(self, lat: float = 53.4, lon: float = -8.2) -> None:
        self.lat = lat
        self.lon = lon

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        return fetch_hourly(
            lat=self.lat,
            lon=self.lon,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        return DataQualityReport(
            missing_pct=0.0,
            duplicates=int(df.index.duplicated().sum()),
            outliers=0,
            gaps=[],
        )
