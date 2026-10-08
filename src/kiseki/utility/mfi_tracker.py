from collections import deque


class MFITracker:
    """
    Memory-efficient, stateful Money Flow Index calculator.
    Designed for real-time streaming without re-looping history.

    Uses a rolling window of `period` directional typical-price money flows
    (requiring `period + 1` candles). Returns 0.0 during warm-up;
    `initialized` indicates readiness. No directional flow yields 50.0.
    """

    def __init__(self, period: int = 14):
        if isinstance(period, bool) or not isinstance(period, int) or period < 1:
            raise ValueError("period must be a positive integer")

        self.period = period
        self.prev_mfi = 0.0
        self.prev_typical_price = None
        self.initialized = False

        # O(period) memory, O(1) updates: Keep only the rolling flow window.
        self._flows = deque()
        self._positive_sum = 0.0
        self._negative_sum = 0.0

    def update(self, high: float, low: float, close: float, volume: float) -> float:
        # 1. Calculate typical price (the first candle is a baseline).
        typical_price = (high + low + close) / 3.0
        if self.prev_typical_price is None:
            self.prev_typical_price = typical_price
            return 0.0

        money_flow = typical_price * volume
        positive_flow = money_flow if typical_price > self.prev_typical_price else 0.0
        negative_flow = money_flow if typical_price < self.prev_typical_price else 0.0
        self.prev_typical_price = typical_price

        # 2. Update rolling sums, removing the oldest flow when necessary.
        if len(self._flows) == self.period:
            old_positive, old_negative = self._flows.popleft()
            self._positive_sum -= old_positive
            self._negative_sum -= old_negative

        self._flows.append((positive_flow, negative_flow))
        self._positive_sum += positive_flow
        self._negative_sum += negative_flow

        if len(self._flows) < self.period:
            return 0.0

        self.initialized = True

        # 3. Convert directional money flows into MFI.
        positive_sum = max(self._positive_sum, 0.0)
        negative_sum = max(self._negative_sum, 0.0)
        total_flow = positive_sum + negative_sum
        self.prev_mfi = 50.0 if total_flow == 0.0 else 100.0 * positive_sum / total_flow

        return self.prev_mfi
