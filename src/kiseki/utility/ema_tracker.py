class EMATracker:
    """
    Memory-efficient, stateful EMA calculator using exponential smoothing.
    Designed for real-time streaming without re-looping history.

    Seeds the EMA with the simple mean of the first `period` closes.
    Returns 0.0 during warm-up; `initialized` indicates readiness.
    """

    def __init__(self, period: int = 14):
        if isinstance(period, bool) or not isinstance(period, int) or period < 1:
            raise ValueError("period must be a positive integer")

        self.period = period
        self.prev_ema = 0.0
        self.initialized = False

        # O(1) memory: Use a running sum instead of storing history.
        self._init_close_sum = 0.0
        self._count = 0
        self._alpha = 2.0 / (period + 1)

    def update(self, close: float) -> float:
        # 1. Initialization Phase (Simple Mean)
        if not self.initialized:
            self._init_close_sum += close
            self._count += 1

            if self._count == self.period:
                self.prev_ema = self._init_close_sum / self.period
                self.initialized = True
                self._init_close_sum = 0.0
                return self.prev_ema

            return 0.0

        # 2. Exponential Smoothing Phase (Streaming)
        # Formula: EMA = Prev EMA + Alpha * (Close - Prev EMA)
        self.prev_ema += self._alpha * (close - self.prev_ema)

        return self.prev_ema
