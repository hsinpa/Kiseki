"""Single-stock candle SMA example: python -m kiseki.example_candlestick.

Edit the explicit parameters below. CSV columns: Date, Open, High, Low,
Close, Volume. Date must label candle completion.
"""

from kiseki import (
    Backtester, CandleStickBroker, CostModel, CandleStickDataLoader, SmaCandleStrategy,
)


def main(files: list[str], symbol: str, initial_capital: float = 1_000_000.0):
    loader = CandleStickDataLoader(csv_pathes=files)
    broker = CandleStickBroker(initial_capital, CostModel())
    strategy = SmaCandleStrategy(broker, fast=5, slow=20)
    result = Backtester(strategy, broker, loader, symbol=symbol).run()
    print(result.summary())
    return result


if __name__ == "__main__":
    main(files=["data/QQQ_15min_avg_30min.csv"], symbol="QQQ", initial_capital=1_000_000.0)
