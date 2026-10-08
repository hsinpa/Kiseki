"""Trade both AAPL and MSFT from every completed candle.

Run: python -m kiseki.example_multistock
Edit the folder literals below. CSV columns: Date, Open, High, Low, Close,
Volume. Date must label completion, not the start of the candle.
"""

from collections import deque

import pandas as pd

from kiseki.utility.tick_aggregator import Candlestick
from kiseki import (
    Backtester, CandleStickBroker, CandleStickDataLoader, CandleStickStrategy,
    CostModel, MultiStockDataLoader, Order, Side, StockSource,
)


class PortfolioTrendStrategy(CandleStickStrategy):
    """Trade each stock long/flat using its SMA and the other stock's candle.

    Histories are separate per symbol. Both equal-time market snapshots are
    already available, regardless of callback order. Missing or stale peer
    data targets flat. Only order fills, not signals, change positions.
    """

    def __init__(self, broker, symbols: tuple[str, str] = ("AAPL", "MSFT"),
                 window: int = 5, position_size: int = 10,
                 max_peer_age: pd.Timedelta = pd.Timedelta("1D")):
        super().__init__(broker)
        if len(symbols) != 2 or len(set(symbols)) != 2:
            raise ValueError("Provide two distinct symbols")
        if window < 2 or position_size <= 0 or max_peer_age < pd.Timedelta(0):
            raise ValueError("Invalid window, position_size or max_peer_age")
        self.symbols = symbols
        self.window = window
        self.position_size = position_size
        self.max_peer_age = max_peer_age
        self._closes: dict[str, deque] = {}

    def on_start(self) -> None:
        self._closes.clear()

    def on_candle(self, symbol: str, candle: Candlestick) -> list[Order]:
        if symbol not in self.symbols:
            return []
        closes = self._closes.setdefault(symbol, deque(maxlen=self.window))
        closes.append(candle.close)
        peer_symbol = next(other for other in self.symbols if other != symbol)
        peer = self.market.snapshots[peer_symbol]
        fresh = peer is not None and candle.date - peer.date <= self.max_peer_age
        bullish = (fresh and len(closes) == self.window
                   and candle.close > sum(closes) / self.window
                   and peer.close > peer.open)
        target = self.position_size if bullish else 0
        delta = target - self.broker.position(symbol)
        if delta:
            return [Order(symbol=symbol, side=Side.BUY if delta > 0 else Side.SELL,
                          qty=abs(delta))]
        return []


def main(initial_capital: float = 1_000_000.0, position_size: int = 10,
         window: int = 5, max_peer_age: pd.Timedelta = pd.Timedelta("1D"),
         force_flatten_at_end: bool = False):
    loader = MultiStockDataLoader(sources=[
        StockSource("QQQ", CandleStickDataLoader(csv_pathes=['data/QQQ_15min_avg_30min.csv'])),
        StockSource("TQQQ", CandleStickDataLoader(csv_pathes=['data/TQQQ_15min_avg_30min.csv'])),
    ])
    broker = CandleStickBroker(initial_capital, CostModel())
    strategy = PortfolioTrendStrategy(broker, window=window,
                                     position_size=position_size,
                                     max_peer_age=max_peer_age)
    result = Backtester(strategy, broker, loader, force_flatten_at_end).run()
    print(result.summary())
    print(result.trades_df())
    return result


if __name__ == "__main__":
    main(
        initial_capital=1_000_000.0,
        position_size=10,
        window=5,
        max_peer_age=pd.Timedelta("1D"),
        force_flatten_at_end=False,
    )
