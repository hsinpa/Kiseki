from collections import deque


class SMATracker:
    """
    Stateful SMA calculator using a rolling window and running sum.
    Designed for real-time streaming without re-looping history.

    Returns 0.0 during warm-up; `initialized` indicates readiness.
    Stores at most `period` closes and updates in O(1) time.
    """

    def __init__(self, period: int = 14):
        if isinstance(period, bool) or not isinstance(period, int) or period < 1:
            raise ValueError("period must be a positive integer")

        self.period = period
        self.prev_sma = 0.0
        self.initialized = False

        self._closes = deque()
        self._close_sum = 0.0

    def update(self, close: float) -> float:
        # 1. Remove the oldest close when the window is full.
        if len(self._closes) == self.period:
            self._close_sum -= self._closes.popleft()

        self._closes.append(close)
        self._close_sum += close

        # 2. Return 0.0 until the first complete window is available.
        if len(self._closes) < self.period:
            return 0.0

        self.initialized = True
        self.prev_sma = self._close_sum / self.period
        return self.prev_sma
