from datetime import date
from typing import Optional

class VWAP_Tool:
    """Tick-derived session VWAP from bars, using O(1) memory.

    Add bars chronologically. Each calendar date starts a new session.
    Before positive session volume is available, use the latest bar's close.
    """

    def __init__(self):
        self.reset()

    def add(self, session_date: date, wap: float, close: float, volume: int) -> None:
        if session_date != self._session_date:
            self.reset()
            self._session_date = session_date

        self._price_volume_sum += wap * volume
        self._vol_sum += volume
        self._last_close = close
        self._bar_count += 1

    def get(self) -> float:
        """Return session VWAP, latest close as fallback, or 0.0 before any bars."""
        if self._vol_sum > 0:
            return self._price_volume_sum / self._vol_sum
        return self._last_close

    def reset(self) -> None:
        self._session_date: Optional[date] = None
        self._price_volume_sum = 0.0
        self._vol_sum = 0.0
        self._last_close = 0.0
        self._bar_count = 0

    @property
    def total_volume(self) -> float:
        return self._vol_sum

    @property
    def bar_count(self) -> int:
        return self._bar_count
