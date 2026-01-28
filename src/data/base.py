from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd


@dataclass
class DataQualityReport:
    missing_pct: float
    duplicates: int
    outliers: int
    gaps: list[tuple[datetime, datetime]]


class DataSource(ABC):
    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        raise NotImplementedError

    def fetch_with_fallback(
        self,
        start: datetime,
        end: datetime,
        fallback_sources: Iterable["DataSource"],
    ) -> pd.DataFrame:
        """Try primary source, fall back to alternatives on failure."""
        try:
            return self.fetch(start, end)
        except Exception as exc:
            last_err = exc
            for source in fallback_sources:
                try:
                    return source.fetch(start, end)
                except Exception as fallback_exc:
                    last_err = fallback_exc
                    continue
            raise RuntimeError(f"All data sources failed: {last_err}") from last_err
