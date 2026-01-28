from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .base import DataQualityReport


@dataclass
class Gap:
    start: datetime
    end: datetime


class PriceDataValidator:
    """Validate electricity price data quality."""

    PRICE_MIN = -500
    PRICE_MAX = 4000

    def __init__(self, df: pd.DataFrame, price_col: str = "dam_eur_mwh") -> None:
        self.df = df
        self.price_col = price_col
        self.issues: list[str] = []

    def check_completeness(self, expected_freq: str = "H") -> float:
        full_idx = pd.date_range(
            self.df.index.min(),
            self.df.index.max(),
            freq=expected_freq,
        )
        missing = len(full_idx) - len(self.df)
        missing_pct = (missing / len(full_idx) * 100) if len(full_idx) else 0.0

        if missing_pct > 5:
            self.issues.append(f"High missing data: {missing_pct:.1f}%")
        return missing_pct

    def check_outliers(self, n_std: float = 4) -> pd.Series:
        prices = self.df[self.price_col]
        mean, std = prices.mean(), prices.std()
        outliers = (prices < mean - n_std * std) | (prices > mean + n_std * std)
        return outliers

    def check_physical_bounds(self) -> pd.Series:
        prices = self.df[self.price_col]
        invalid = (prices < self.PRICE_MIN) | (prices > self.PRICE_MAX)
        if invalid.any():
            self.issues.append(f"Physical bounds violated: {invalid.sum()} rows")
        return invalid

    def check_stale_data(self, max_constant_hours: int = 6) -> list[tuple[int, int]]:
        prices = self.df[self.price_col]
        stale_periods: list[tuple[int, int]] = []

        constant_count = 0
        for i in range(1, len(prices)):
            if prices.iloc[i] == prices.iloc[i - 1]:
                constant_count += 1
            else:
                if constant_count >= max_constant_hours:
                    stale_periods.append((i - constant_count, i))
                constant_count = 0

        if constant_count >= max_constant_hours:
            stale_periods.append((len(prices) - constant_count, len(prices)))

        return stale_periods

    def detect_gaps(self, expected_freq: str = "H") -> list[tuple[datetime, datetime]]:
        if self.df.empty:
            return []
        full_idx = pd.date_range(
            self.df.index.min(),
            self.df.index.max(),
            freq=expected_freq,
        )
        missing = full_idx.difference(self.df.index)
        if missing.empty:
            return []

        gaps = []
        gap_start = missing[0]
        prev = missing[0]
        for ts in missing[1:]:
            if (ts - prev) > pd.Timedelta(expected_freq):
                gaps.append((gap_start.to_pydatetime(), prev.to_pydatetime()))
                gap_start = ts
            prev = ts
        gaps.append((gap_start.to_pydatetime(), prev.to_pydatetime()))
        return gaps

    def full_report(self) -> dict:
        return {
            "missing_pct": self.check_completeness(),
            "outliers": int(self.check_outliers().sum()),
            "invalid_bounds": int(self.check_physical_bounds().sum()),
            "stale_periods": len(self.check_stale_data()),
            "issues": self.issues,
        }


def quality_report_from_validator(validator: PriceDataValidator) -> DataQualityReport:
    report = validator.full_report()
    gaps = validator.detect_gaps()
    duplicates = int(validator.df.index.duplicated().sum())
    return DataQualityReport(
        missing_pct=report["missing_pct"],
        duplicates=duplicates,
        outliers=report["outliers"],
        gaps=gaps,
    )
