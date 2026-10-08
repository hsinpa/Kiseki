import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

from kiseki.types import Side, Order, BacktestResult, BuyAndHoldResult, MarketEvent
from kiseki.broker import BaseBroker
from kiseki.metrics import build_result
from kiseki.validation import validate_market_event


class BacktestSession:
    """Portfolio execution shared by the strategy and RL drivers.

    A batch fills earlier orders first, then marks all updated symbols. New
    candle orders wait for the target symbol's next batch, not the next event
    of another symbol. Post-trade equity is recorded once per batch. Results
    prepend initial capital at the first batch timestamp, before its recorded
    equity, so initial losses count without inventing a prior trading day.
    Unfilled orders are dropped at end of data; optional liquidation uses each
    symbol's last event.
    Orders execute sequentially: pending fills in sorted-symbol/source order,
    then new orders in callback emission order. Each sees remaining cash and
    holdings after preceding fills; no cash is reserved at signal time.
    """

    def __init__(self, broker: BaseBroker, force_flatten_at_end: bool = False,
                 symbols: Sequence[str] = ()):
        self.broker = broker
        self.force_flatten_at_end = force_flatten_at_end
        self.symbols = frozenset(symbols)
        self.timestamps: list = []
        self.equity: list[float] = []
        self._events: dict[str, MarketEvent] = {}
        self._last_events: dict[str, MarketEvent] = {}
        self._first_events: dict[str, MarketEvent] = {}
        self._pending: dict[str, list[Order]] = defaultdict(list)
        self._when = None

    def begin_batch(self, events: Mapping[str, MarketEvent]) -> None:
        if not events:
            raise ValueError("An event batch must not be empty")
        current_events = dict(sorted(events.items()))
        if set(events) - self.symbols:
            raise ValueError("Event batch contains an unconfigured symbol")
        times = [self.broker.event_time(event) for event in current_events.values()]
        if len({self.broker._to_ns(time) for time in times}) != 1:
            raise ValueError("All events in a batch must share a timestamp")
        # Validate the entire batch before consuming pending orders, financing,
        # or marks. A bad symbol must not leave another symbol partly executed.
        for symbol, event in current_events.items():
            validate_market_event(event, f"Symbol {symbol}")
        self._events = current_events
        self._when = times[0]
        self.broker.accrue_financing(self._when)
        # Close prices must not affect trades exited at this candle's open.
        for symbol, event in self._events.items():
            for order in self._pending.pop(symbol, []):
                self.broker.execute(order, event)
        for symbol, event in self._events.items():
            self.broker.mark(symbol, event)
        for symbol, event in self._events.items():
            self._first_events.setdefault(symbol, event)
        self._last_events.update(self._events)

    def end_batch(self, orders: list[Order]) -> None:
        if not isinstance(orders, list):
            raise TypeError("Strategy callbacks must return list[Order]; use [] to hold")
        for order in orders:
            if not isinstance(order, Order):
                raise TypeError("Orders must contain Order objects")
            if order.symbol not in self.symbols:
                raise ValueError(f"Unknown order symbol: {order.symbol}")
            if not isinstance(order.side, Side):
                raise ValueError("Order side must be BUY or SELL")
            if not isinstance(order.qty, int) or isinstance(order.qty, bool) or order.qty <= 0:
                raise ValueError("Order quantity must be a positive integer")
            if not self.broker.defers_fill and order.symbol not in self._events:
                raise ValueError(f"No current quote for {order.symbol}; cannot fill a stale tick")
        for order in orders:
            if self.broker.defers_fill:
                self._pending[order.symbol].append(order)
            else:
                self.broker.execute(order, self._events[order.symbol])
        self.timestamps.append(self._when)
        self.equity.append(self.broker.equity)

    def finish(self) -> BacktestResult:
        self._pending.clear()
        if self.force_flatten_at_end:
            for symbol, event in sorted(self._last_events.items()):
                qty = self.broker.position(symbol)
                if qty:
                    side = Side.SELL if qty > 0 else Side.BUY
                    self.broker.execute(
                        Order(symbol=symbol, side=side, qty=abs(qty),
                              reason="End-of-data liquidation"), event,
                        price=self.broker.liquidation_price(event, side),
                    )
            if self.equity:
                self.equity[-1] = self.broker.equity
        result = build_result(self.timestamps, self.equity, self.broker.trades,
                              self.broker.initial_capital)
        result.rejected_orders = list(self.broker.rejected_orders)
        capital = self.broker.initial_capital
        for symbol, first in sorted(self._first_events.items()):
            last = self._last_events[symbol]
            entry, exit = float(first.close), float(last.close)
            valid = all(math.isfinite(price) and price > 0 for price in (entry, exit))
            change = exit / entry - 1 if valid else math.nan
            result.buy_and_hold[symbol] = BuyAndHoldResult(
                symbol=symbol, initial_capital=capital,
                entry_price=entry, exit_price=exit,
                entry_time=self.broker.event_time(first),
                exit_time=self.broker.event_time(last),
                revenue=capital * change,
                return_pct=change * 100 if capital else (0.0 if valid else math.nan),
            )
        return result
