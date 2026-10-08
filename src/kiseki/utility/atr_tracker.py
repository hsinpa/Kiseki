class ATRTracker:
    """
    Memory-efficient, stateful ATR calculator using Wilder's Smoothing.
    Designed for real-time streaming without re-looping history.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.prev_atr = 0.0
        self.prev_close = 0.0
        self.initialized = False

        # O(1) Memory efficiency: Use running sum instead of storing a list
        self._init_tr_sum = 0.0
        self._count = 0

    def update(self, high: float, low: float, open: float, close: float) -> float:
        # 1. Calculate True Range (TR)
        if self._count == 0:
            # First candle ever: TR is just High - Low
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )

        # Update state for next iteration
        self.prev_close = close

        # 2. Initialization Phase (Simple Mean)
        if not self.initialized:
            self._init_tr_sum += tr
            self._count += 1

            # Check if we have enough data to seed the ATR
            if self._count == self.period:
                self.prev_atr = self._init_tr_sum / self.period
                self.initialized = True
                # Optional: Free up memory (though floats take negligible space)
                self._init_tr_sum = 0.0
                return self.prev_atr
            else:
                # Not enough data yet
                return 0.0

        # 3. Wilder's Smoothing Phase (Streaming)
        # Formula: ATR = ((Prev ATR * (Period - 1)) + TR) / Period
        self.prev_atr = (self.prev_atr * (self.period - 1) + tr) / self.period

        return self.prev_atr