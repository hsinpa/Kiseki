import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from kiseki.utility.tick_aggregator import Tick, Candlestick
from kiseki.types import Side, Order, RejectedOrder, Trade, Timestamp
from kiseki.validation import validate_market_event

# Calendar-day basis for annualizing the short-borrow financing rate.
SECONDS_PER_YEAR = 365 * 24 * 3600

@dataclass
class CostModel:
    """Universal trading-cost model. Every fee is a parameter; defaults to 0.0
    so unused components drop out. Costs are priced from `notional` and `shares`
    together, so per-share venues (US) and per-notional venues (TW) both work
    without assuming a fixed price.

    Commission applies to BOTH legs. Levies are split into buy-leg and sell-leg
    (TW transaction tax = sell; UK stamp duty = buy; US SEC+TAF = sell).
    `shares` is only consulted when a per-share rate is non-zero.
    """
    # Commission (both legs)
    commission_rate:      float = 0.0   # fraction of notional
    commission_per_share: float = 0.0   # currency per share
    min_commission:       float = 0.0   # per-order floor
    max_commission_rate:  float = 0.0   # cap as fraction of notional (0 = none)

    # Sell-leg levies (taxes / regulatory)
    sell_rate:           float = 0.0    # fraction of notional
    sell_per_share:      float = 0.0    # currency per share
    sell_per_share_cap:  float = 0.0    # per-order cap (0 = none)

    # Buy-leg levies
    buy_rate:            float = 0.0
    buy_per_share:       float = 0.0
    buy_per_share_cap:   float = 0.0

    # Short carry
    borrow_rate:         float = 0.0    # annualized

    def _commission(self, notional: float, shares: float) -> float:
        c = notional * self.commission_rate + shares * self.commission_per_share
        c = max(c, self.min_commission)
        if self.max_commission_rate > 0.0:                 # floor first, cap last
            c = min(c, notional * self.max_commission_rate)  # cap wins per IBKR rule
        return c

    @staticmethod
    def _levy(notional, shares, rate, per_share, cap) -> float:
        ps = shares * per_share
        if cap > 0.0:
            ps = min(ps, cap)
        return notional * rate + ps

    def cost(self, notional: float, side: Side, shares: float = 0.0) -> float:
        c = self._commission(notional, shares)
        if side is Side.BUY:
            c += self._levy(notional, shares, self.buy_rate,
                            self.buy_per_share, self.buy_per_share_cap)
        else:  # SELL
            c += self._levy(notional, shares, self.sell_rate,
                            self.sell_per_share, self.sell_per_share_cap)
        return c

    def carry_cost(self, notional: float, elapsed_seconds: float) -> float:
        if self.borrow_rate <= 0.0 or elapsed_seconds <= 0.0:
            return 0.0
        return notional * self.borrow_rate * (elapsed_seconds / SECONDS_PER_YEAR)

    # ---- venue presets: the only place numbers are hardcoded ----
    @classmethod
    def taiwan_day_trade(cls, commission_rate: float = 0.001425) -> "CostModel":
        return cls(commission_rate=commission_rate, sell_rate=0.0015)

    @classmethod
    def taiwan_overnight(cls, commission_rate: float = 0.001425) -> "CostModel":
        return cls(commission_rate=commission_rate, sell_rate=0.003)

    @classmethod
    def ibkr_pro_fixed(cls, sec_fee_rate: float = 0.0000206) -> "CostModel":
        return cls(commission_per_share=0.005, min_commission=1.00,
                   max_commission_rate=0.01, sell_rate=sec_fee_rate,
                   sell_per_share=0.000195, sell_per_share_cap=9.79)

    @classmethod
    def ibkr_lite(cls, sec_fee_rate: float = 0.0000206) -> "CostModel":
        return cls(sell_rate=sec_fee_rate,
                   sell_per_share=0.000195, sell_per_share_cap=9.79)



@dataclass
class Position:
    """Average-cost accounting state for one instrument.

    High/low water marks span the continuous holding period until fully flat.
    With scale-ins, excursions use the current average entry price, not lots.
    """
    qty: int = 0
    avg_entry_price: float = 0.0
    mark_price: float = 0.0
    financing_accrued: float = 0.0
    entry_cost_per_unit: float = 0.0
    entry_time: Optional[Timestamp] = None
    high_water: float = 0.0
    low_water: float = 0.0


class BaseBroker:
    """Long-only cash broker with shared cash and independent positions.

    Buys fill fully only when cash covers price plus fees; otherwise they are
    rejected. Sells are clamped to current holdings before filling fully.
    Liquidity is assumed sufficient: no partial market fills or market impact.
    Orders share cash and execute sequentially in driver order.
    Average-cost realized-PnL accounting, configurable costs, equity and trades.

    Subclasses supply only the data-source-specific bits:
      - `_fill_price(event, side)`: the price an order fills at.
      - `event_time(event)`: the event's timestamp (`int` or `pd.Timestamp`).
      - `liquidation_price(event, side)`: price for the end-of-data forced exit.
      - `defers_fill`: whether the engine should defer an order to the next event
        (next-bar execution) rather than filling it on the signal event.
    Everything else is identical across data sources.
    """

    #: When True, the engine executes an order on the NEXT event, not the one
    #: that produced it (next-bar execution, to avoid look-ahead).
    defers_fill: bool = False

    def __init__(self, initial_capital: float, cost_model: Optional[CostModel] = None):
        if not math.isfinite(initial_capital) or initial_capital < 0:
            raise ValueError("initial_capital must be finite and nonnegative")
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()

        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.rejected_orders: List[RejectedOrder] = []
        self.financing_paid: float = 0.0
        self._last_time: Optional[Timestamp] = None

    def reset(self) -> None:
        """Restore the initial account without changing capital or fee settings.

        Replace containers rather than clearing them, so earlier results keep
        their trade records and any retained position references stay intact.
        """
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.rejected_orders = []
        self.financing_paid = 0.0
        self._last_time = None

    def position(self, symbol: str) -> int:
        """Signed filled quantity; reading an unseen symbol returns zero."""
        state = self.positions.get(symbol)
        return state.qty if state is not None else 0

    # ---------- per-source hooks ----------

    def _fill_price(self, event, side: Side) -> float:
        raise NotImplementedError

    def event_time(self, event) -> Timestamp:
        raise NotImplementedError

    def liquidation_price(self, event, side: Side) -> float:
        """Price for the engine's end-of-data forced flatten. Defaults to the
        normal fill price; overridden where the final exit differs (e.g. a
        candle has no next bar, so it liquidates at its close, not its open)."""
        return self._fill_price(event, side)

    # ---------- marking ----------

    @staticmethod
    def _to_ns(t: Timestamp) -> int:
        # Tick.ts is int epoch-ns; Candlestick.date is a pd.Timestamp.
        return t.value if isinstance(t, pd.Timestamp) else int(t)

    def accrue_financing(self, when: Timestamp) -> None:
        """Accrue every short over portfolio time, before this batch's fills."""
        if self._last_time is not None:
            elapsed = (self._to_ns(when) - self._to_ns(self._last_time)) / 1e9
            if elapsed < 0:
                raise ValueError("Portfolio timestamps must be nondecreasing")
            for state in self.positions.values():
                if state.qty < 0:
                    carry = self.cost_model.carry_cost(abs(state.qty) * state.mark_price, elapsed)
                    self.cash -= carry
                    self.financing_paid += carry
                    state.financing_accrued += carry
        self._last_time = when

    def mark(self, symbol: str, event) -> None:
        """Mark equity at close and track the held event's price extremes.

        Candle orders fill at open before marking: new positions include this
        bar's high/low, while positions fully exited at open exclude them.
        Remaining shares after a partial exit still experience this bar.
        """
        self.event_time(event)  # Enforce this broker's event type, even without orders.
        validate_market_event(event, f"Symbol {symbol}")
        state = self.positions.setdefault(symbol, Position())
        state.mark_price = event.close
        if state.qty != 0:
            high = event.high if isinstance(event, Candlestick) else state.mark_price
            low = event.low if isinstance(event, Candlestick) else state.mark_price
            state.high_water = max(state.high_water, high)
            state.low_water = min(state.low_water, low)

    @property
    def equity(self) -> float:
        return self.cash + sum(state.qty * state.mark_price
                               for state in self.positions.values())

    # ---------- execution ----------

    def affordable_buy_quantity(self, price: float, requested: int) -> int:
        """Sizing helper for strategies, not permission for a partial buy fill.

        Execution independently checks the full request at its actual price.
        """
        if price <= 0 or requested <= 0 or self.cash <= 0:
            return 0
        low, high = 0, min(requested, int(self.cash // price))
        while low < high:
            qty = (low + high + 1) // 2
            total = price * qty + self.cost_model.cost(price * qty, Side.BUY, qty)
            if total <= self.cash:
                low = qty
            else:
                high = qty - 1
        return low

    def execute(self, order: Order, event, price: Optional[float] = None) -> None:
        if not isinstance(order.side, Side):
            raise ValueError("Order side must be BUY or SELL")
        if not isinstance(order.qty, int) or isinstance(order.qty, bool) or order.qty <= 0:
            raise ValueError("Order quantity must be a positive integer")
        qty = order.qty
        when = self.event_time(event)
        validate_market_event(event, f"Symbol {order.symbol}")

        # `price` lets the engine override the fill (used for the end-of-data
        # liquidation); otherwise use the broker's normal fill rule.
        fill_price = price if price is not None else self._fill_price(event, order.side)
        if not math.isfinite(fill_price) or fill_price <= 0:
            raise ValueError("Execution price must be finite and positive")
        if order.side is Side.SELL:
            qty = min(qty, max(self.position(order.symbol), 0))
            if qty == 0:
                self.rejected_orders.append(RejectedOrder(
                    order.symbol, order.side, order.qty, when, fill_price,
                    "No shares owned", order.reason,
                ))
                return
        signed_delta = qty if order.side is Side.BUY else -qty
        notional = fill_price * qty
        fill_cost = self.cost_model.cost(notional, order.side, shares=qty)
        if not math.isfinite(fill_cost) or fill_cost < 0:
            raise ValueError("Execution fees must be finite and nonnegative")
        if order.side is Side.BUY and notional + fill_cost > self.cash:
            self.rejected_orders.append(RejectedOrder(
                order.symbol, order.side, order.qty, when, fill_price,
                "Insufficient cash including fees", order.reason,
            ))
            return
        state = self.positions.setdefault(order.symbol, Position())

        # Cash: pay price on buy, receive on sell; always pay costs.
        self.cash -= fill_price * signed_delta
        self.cash -= fill_cost

        old_pos = state.qty
        new_pos = old_pos + signed_delta
        # Include the actual exit price in the old position's excursions.
        if old_pos:
            state.high_water = max(state.high_water, fill_price)
            state.low_water = min(state.low_water, fill_price)

        opening_or_adding = old_pos == 0 or (old_pos > 0) == (signed_delta > 0)
        if opening_or_adding:
            if old_pos == 0:
                state.entry_time = when
                state.high_water = state.low_water = fill_price
            new_abs = abs(new_pos)
            state.avg_entry_price = (
                state.avg_entry_price * abs(old_pos) + fill_price * qty
            ) / new_abs
            state.entry_cost_per_unit = (
                state.entry_cost_per_unit * abs(old_pos) + fill_cost
            ) / new_abs
            state.qty = new_pos
            return

        closing_qty = min(qty, abs(old_pos))
        direction = 1 if old_pos > 0 else -1
        gross = (fill_price - state.avg_entry_price) * direction * closing_qty
        entry_costs = state.entry_cost_per_unit * closing_qty
        exit_costs = fill_cost * (closing_qty / qty)
        financing = state.financing_accrued * (closing_qty / abs(old_pos))
        state.financing_accrued -= financing
        if direction == 1:
            mfe = max(0.0, state.high_water - state.avg_entry_price) * closing_qty
            mae = min(0.0, state.low_water - state.avg_entry_price) * closing_qty
        else:
            mfe = max(0.0, state.avg_entry_price - state.low_water) * closing_qty
            mae = min(0.0, state.avg_entry_price - state.high_water) * closing_qty
        self.trades.append(Trade(
            symbol=order.symbol, direction=direction, qty=closing_qty,
            entry_price=state.avg_entry_price, exit_price=fill_price,
            entry_time=state.entry_time, exit_time=when,
            costs=entry_costs + exit_costs, financing=financing,
            pnl=gross - entry_costs - exit_costs - financing, mfe=mfe, mae=mae,
            exit_reason=order.reason,
        ))
        state.qty = new_pos
        if new_pos == 0:
            state.avg_entry_price = state.entry_cost_per_unit = 0.0
            state.financing_accrued = 0.0
            state.high_water = state.low_water = 0.0
            state.entry_time = None
        elif (new_pos > 0) != (old_pos > 0):
            state.avg_entry_price = fill_price
            state.entry_cost_per_unit = fill_cost / qty
            state.entry_time = when
            state.financing_accrued = 0.0
            state.high_water = state.low_water = fill_price


class TickBroker(BaseBroker):
    """Broker for a Tick stream: fills cross the spread (BUY @ ask, SELL @ bid)."""

    def _fill_price(self, event: Tick, side: Side) -> float:
        return event.ask_price if side is Side.BUY else event.bid_price

    def event_time(self, event: Tick) -> Timestamp:
        if not isinstance(event, Tick):
            raise TypeError("TickBroker requires Tick events for every symbol")
        return event.ts


class CandleStickBroker(BaseBroker):
    """Broker for a Candlestick stream.

    Candles carry no bid/ask, so orders fill at the bar `open`. Combined with
    the engine's next-bar execution (`defers_fill = True`), a signal generated
    on bar t fills at bar t+1's open — never the same close it was computed
    from — which avoids look-ahead. The end-of-data forced flatten has no next
    bar, so it liquidates at the final bar's `close` (its mark price).
    """

    defers_fill = True

    def _fill_price(self, event: Candlestick, side: Side) -> float:
        return event.open

    def event_time(self, event: Candlestick) -> Timestamp:
        if not isinstance(event, Candlestick):
            raise TypeError("CandleStickBroker requires Candlestick events for every symbol")
        return event.date

    def liquidation_price(self, event: Candlestick, side: Side) -> float:
        return event.close


# Backwards-compatible alias: the original tick-only broker.
SimBroker = TickBroker
