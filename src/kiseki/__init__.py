"""Simple event-driven, long-only cash backtesting platform.

Buys fill completely only if execution-time cash covers price plus fees.
Sells are clamped to holdings before filling; short selling is unsupported.
Liquidity is assumed sufficient, with no partial market fills or market impact.
Rejected orders are available on broker.rejected_orders and result.rejected_orders.

Supports two data sources sharing one engine, accounting, and metrics:
    # Tick stream (bid/ask, spread-crossing fills)
    from kiseki import Backtester, TickBroker, CostModel, TickDataLoader, SmaCrossStrategy
    # Candlestick stream (OHLCV, fills at next bar's open)
    from kiseki import Backtester, CandleStickBroker, CandleStickDataLoader, SmaCandleStrategy

    from kiseki import Strategy, TickStrategy, CandleStickStrategy
    from kiseki import Order, Side, Trade, BacktestResult
"""

from kiseki.types import Side, Order, RejectedOrder, Trade, BacktestResult, BuyAndHoldResult, MarketContext
from kiseki.data import (
    BaseDataLoader, TickDataLoader, CandleStickDataLoader, DataLoader,
    StockSource, MultiStockDataLoader,
)
from kiseki.broker import CostModel, Position, BaseBroker, TickBroker, CandleStickBroker, SimBroker
from kiseki.strategy import (
    Strategy,
    TickStrategy,
    CandleStickStrategy,
    SmaCrossStrategy,
    SmaCandleStrategy,
)
from kiseki.session import BacktestSession
from kiseki.engine import Backtester

__all__ = [
    "Side",
    "Order",
    "RejectedOrder",
    "Trade",
    "BacktestResult",
    "BuyAndHoldResult",
    "MarketContext",
    "StockSource",
    "MultiStockDataLoader",
    # data loaders
    "BaseDataLoader",
    "TickDataLoader",
    "CandleStickDataLoader",
    "DataLoader",
    # brokers
    "CostModel",
    "Position",
    "BaseBroker",
    "TickBroker",
    "CandleStickBroker",
    "SimBroker",
    # strategies
    "Strategy",
    "TickStrategy",
    "CandleStickStrategy",
    "SmaCrossStrategy",
    "SmaCandleStrategy",
    # engine
    "BacktestSession",
    "Backtester",
]
