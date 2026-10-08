from types import MappingProxyType

from kiseki.types import BacktestResult, BacktestDataStream, MarketContext
from kiseki.broker import BaseBroker
from kiseki.data import MultiStockDataLoader, StockSource
from kiseki.strategy import Strategy
from kiseki.session import BacktestSession


class Backtester:
    """Dispatch every tradable symbol in chronological, sorted-symbol batches.

    For raw single-stock loaders, supply symbol explicitly. Multi-stock loaders
    supply identities themselves. All equal-time snapshots and fills are visible
    before any callback; callback orders are routed only after the whole batch.
    Call reset() explicitly before reusing this instance for an independent run.
    The loader must be replayable; reset does not rewind one-shot iterators.
    """

    def __init__(self, strategy: Strategy, broker: BaseBroker, loader: BacktestDataStream,
                 force_flatten_at_end: bool = False, symbol: str | None = None):
        self.strategy = strategy
        self.broker = broker
        self.loader = loader
        self.force_flatten_at_end = force_flatten_at_end
        self.symbol = symbol
        sources = getattr(loader, "sources", None)
        if sources is None:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("A raw single-stock loader requires symbol")
            self.symbols = (symbol,)
            self.loader = MultiStockDataLoader([StockSource(symbol, loader)])
        else:
            self.symbols = tuple(source.symbol for source in sources)
        if strategy.broker is not broker:
            raise ValueError("Strategy and Backtester must share the same broker")

    def reset(self) -> None:
        """Restore account and strategy state, preserving configuration/results.

        on_start() must reset custom strategy history and be safe to call more
        than once: it runs here and again at the beginning of each replay.
        Session state (including pending orders) is local to each run.
        """
        self.broker.reset()
        self.strategy.market = None
        self.strategy.position = 0
        self.strategy.on_start()

    def run(self) -> BacktestResult:
        session = BacktestSession(self.broker, self.force_flatten_at_end, self.symbols)
        self.strategy.market = None
        self.strategy.position = 0
        self.strategy.on_start()
        for item in self.loader.stream():
            if isinstance(item, MarketContext):
                context = item
            else:
                events = MappingProxyType({self.symbol: item})
                context = MarketContext(events, events)
            session.begin_batch(context.events)
            self.strategy.market = context
            orders = []
            for symbol, event in sorted(context.events.items()):
                self.strategy.position = self.broker.position(symbol)
                emitted = self.strategy.handle(symbol, event)
                if not isinstance(emitted, list):
                    raise TypeError("Strategy callbacks must return list[Order]; use [] to hold")
                orders.extend(emitted)
            session.end_batch(orders)
        result = session.finish()
        self.strategy.on_finish()
        return result
