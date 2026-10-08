from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from kiseki.utility.tick_aggregator import Tick, Candlestick, TickAggregator
from kiseki.types import Side, Order, MarketContext

if TYPE_CHECKING:
    from kiseki.broker import BaseBroker


class Strategy(ABC):
    """Base strategy with a shared portfolio broker and synchronized market.

    position is the filled quantity of the current callback's symbol. Query
    broker.position(symbol) for any other stock, or broker.positions[symbol]
    for its full accounting state. Custom constructors must call super().__init__.
    market.snapshots contains all latest observations, including equal-time
    updates, before callbacks run. Treat market events and broker state as read-only.
    """

    position: int = 0
    market: MarketContext | None = None

    def __init__(self, broker: "BaseBroker") -> None:
        self.broker = broker

    def on_start(self) -> None:
        """Initialize/reset all custom replay state, preserving configuration.

        Called before each replay and by Backtester.reset(). Implementations
        must be idempotent and clear histories, indicators, and pending flags.
        """

    def on_finish(self) -> None:
        """Called after end-of-data settlement."""

    @abstractmethod
    def handle(self, symbol: str, event) -> list[Order]:
        raise NotImplementedError


class TickStrategy(Strategy):
    def handle(self, symbol: str, event: Tick) -> list[Order]:
        return self.on_tick(symbol, event)

    @abstractmethod
    def on_tick(self, symbol: str, tick: Tick) -> list[Order]:
        """Return symbol-specific orders; [] means hold."""
        raise NotImplementedError


class CandleStickStrategy(Strategy):
    def handle(self, symbol: str, event: Candlestick) -> list[Order]:
        return self.on_candle(symbol, event)

    @abstractmethod
    def on_candle(self, symbol: str, candle: Candlestick) -> list[Order]:
        """Called for every tradable candle; orders may target other symbols."""
        raise NotImplementedError


def _sma_target_orders(symbol: str, closes: deque, fast: int, slow: int,
                       position: int) -> list[Order]:
    if len(closes) < slow:
        return []
    series = list(closes)
    fast_ma = sum(series[-fast:]) / fast
    slow_ma = sum(series) / slow
    target = 1 if fast_ma > slow_ma else 0
    delta = target - position
    if delta:
        return [Order(symbol, Side.BUY if delta > 0 else Side.SELL, abs(delta))]
    return []


class SmaCrossStrategy(TickStrategy):
    """Long/flat SMA crossover with independent aggregators per stock."""

    def __init__(self, broker, period: str = "1min", fast: int = 5, slow: int = 20):
        super().__init__(broker)
        if not 0 < fast < slow:
            raise ValueError("windows must satisfy 0 < fast < slow")
        self.period = period
        self.fast = fast
        self.slow = slow
        self._aggregators: dict[str, TickAggregator] = {}
        self._closes: dict[str, deque] = {}

    def on_start(self) -> None:
        self._aggregators.clear()
        self._closes.clear()

    def on_tick(self, symbol: str, tick: Tick) -> list[Order]:
        if symbol not in self._aggregators:
            self._aggregators[symbol] = TickAggregator(self.period)
            self._closes[symbol] = deque(maxlen=self.slow)
        bar = self._aggregators[symbol].update(tick)
        if bar is None:
            return []
        self._closes[symbol].append(bar.close)
        return _sma_target_orders(symbol, self._closes[symbol], self.fast,
                                  self.slow, self.position)


class SmaCandleStrategy(CandleStickStrategy):
    """Long/flat SMA crossover with independent candle history per stock."""

    def __init__(self, broker, fast: int = 5, slow: int = 20):
        super().__init__(broker)
        if not 0 < fast < slow:
            raise ValueError("windows must satisfy 0 < fast < slow")
        self.fast = fast
        self.slow = slow
        self._closes: dict[str, deque] = {}

    def on_start(self) -> None:
        self._closes.clear()

    def on_candle(self, symbol: str, candle: Candlestick) -> list[Order]:
        closes = self._closes.setdefault(symbol, deque(maxlen=self.slow))
        closes.append(candle.close)
        return _sma_target_orders(symbol, closes, self.fast, self.slow, self.position)
