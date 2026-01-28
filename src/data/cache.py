from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class CacheResult:
    hit: bool
    data: Optional[pd.DataFrame]
    age: Optional[timedelta]


class DataCache:
    """Lightweight parquet cache with optional TTL."""

    def __init__(self, cache_dir: Path, ttl: timedelta | None = None) -> None:
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def get(self, key: str) -> CacheResult:
        path = self._path(key)
        if not path.exists():
            return CacheResult(hit=False, data=None, age=None)

        modified = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - modified
        if self.ttl is not None and age > self.ttl:
            return CacheResult(hit=False, data=None, age=age)

        df = pd.read_parquet(path)
        return CacheResult(hit=True, data=df, age=age)

    def set(self, key: str, df: pd.DataFrame) -> Path:
        path = self._path(key)
        df.to_parquet(path)
        return path
