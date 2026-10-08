class RSITracker:
    """
    Memory-efficient, stateful RSI calculator using Wilder's Smoothing.
    Designed for real-time streaming without re-looping history.

    Seeds average gains and losses from the first `period` price changes
    (requiring `period + 1` closes). Returns 0.0 during warm-up;
    `initialized` indicates readiness. Unchanged prices yield 50.0.
    """

    def __init__(self, period: int = 14):
        if isinstance(period, bool) or not isinstance(period, int) or period < 1:
            raise ValueError("period must be a positive integer")

        self.period = period
        self.prev_rsi = 0.0
        self.prev_close = None
        self.initialized = False

        # O(1) memory: Use running sums instead of storing history.
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._count = 0

    def update(self, close: float) -> float:
        # 1. Calculate the price change (the first close is a baseline).
        if self.prev_close is None:
            self.prev_close = close
            return 0.0

        change = close - self.prev_close
        self.prev_close = close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        # 2. Initialization Phase (Simple Mean)
        if not self.initialized:
            self._avg_gain += gain
            self._avg_loss += loss
            self._count += 1

            if self._count < self.period:
                return 0.0

            self._avg_gain /= self.period
            self._avg_loss /= self.period
            self.initialized = True
        else:
            # 3. Wilder's Smoothing Phase (Streaming)
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        # 4. Convert smoothed gains and losses into RSI.
        if self._avg_gain == 0.0 and self._avg_loss == 0.0:
            self.prev_rsi = 50.0
        elif self._avg_loss == 0.0:
            self.prev_rsi = 100.0
        else:
            self.prev_rsi = 100.0 - 100.0 / (1.0 + self._avg_gain / self._avg_loss)

        return self.prev_rsi
