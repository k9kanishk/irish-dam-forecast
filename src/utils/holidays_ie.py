from __future__ import annotations
import holidays

_ie = holidays.Ireland()


def is_ie_holiday(d) -> bool:
    return d in _ie
