from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.base import DataQualityReport, DataSource
from src.data.semopx_api import fetch_dam_hrp_recent
from src.data.validators import PriceDataValidator, quality_report_from_validator


class SemopxDayAheadPriceSource(DataSource):
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        days = max(1, (end - start).days + 1)
        df = fetch_dam_hrp_recent(days=days)
        if "ts_utc" in df.columns:
            df = df.set_index("ts_utc")
        df = df.sort_index()
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask]

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        validator = PriceDataValidator(df)
        return quality_report_from_validator(validator)
