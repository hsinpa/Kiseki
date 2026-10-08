# Kiseki

A Python stock backtesting tool for single stocks and shared multi-stock portfolios.

## Install

Requires Python 3.14+, [uv](https://docs.astral.sh/uv/), and Git.
Install into your uv project directly from GitHub:

```sh
uv add "kiseki @ git+https://github.com/hsinpa/Kiseki.git"
```

To work on Kiseki itself:

```sh
git clone https://github.com/hsinpa/Kiseki.git
cd Kiseki
uv sync
```

## Data

Bring your own candle CSVs with columns `Date, Open, High, Low, Close, Volume`.
`Date` must mark candle completion, with unique, increasing timestamps per stock.
Kiseki assumes your data uses the same timeframe and time basis across stocks; it does not resample data.

## Single stock

```python
from kiseki import Backtester, CandleStickBroker, CandleStickDataLoader, SmaCandleStrategy

loader = CandleStickDataLoader(csv_pathes=["data/AAPL.csv"])
broker = CandleStickBroker(initial_capital=10_000)
strategy = SmaCandleStrategy(broker, fast=5, slow=20)

result = Backtester(strategy, broker, loader, symbol="AAPL").run()
print(result.summary())
```

## Multiple stocks

Use one CSV loader per stock. The stocks share the same broker and capital;
this SMA strategy tracks each stock independently.

```python
from kiseki import (
    Backtester, CandleStickBroker, CandleStickDataLoader,
    MultiStockDataLoader, SmaCandleStrategy, StockSource,
)

loader = MultiStockDataLoader([
    StockSource(symbol, CandleStickDataLoader(csv_pathes=[f"data/{symbol}.csv"]))
    for symbol in ["AAPL", "MSFT"]
])
broker = CandleStickBroker(initial_capital=10_000)
strategy = SmaCandleStrategy(broker, fast=5, slow=20)

result = Backtester(strategy, broker, loader).run()
print(result.summary())
```

Use `result.trades_df()` to inspect trades. Positions remain open at the end by
default; pass `force_flatten_at_end=True` to `Backtester` to close them.

More examples: [examples/](examples/).
