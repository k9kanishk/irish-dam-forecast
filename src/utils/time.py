from __future__ import annotations
from zoneinfo import ZoneInfo
from datetime import datetime

IE_TZ = ZoneInfo("Europe/Dublin")


def now_ie() -> datetime:
    return datetime.now(IE_TZ)
