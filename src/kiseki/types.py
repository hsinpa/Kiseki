from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Mapping, Protocol, Union

from kiseki.utility.tick_aggregator import Tick, Candlestick

import pandas as pd

# A timestamp from either data source: ticks carry an int epoch-ns (`ts`),
# candlesticks carry a `pd.Timestamp` (`date`).
Timestamp = Union[int, pd.Timestamp]
MarketEvent = Union[Tick, Candlestick]


@dataclass(frozen=True)
class MarketContext:
    """One timestamp's events and latest observations for every symbol.

    Mappings are independent read-only snapshots. Events are completion-labeled;
    snapshots contain no observations later than this batch's timestamp.
    """
    events: Mapping[str, MarketEvent]
    snapshots: Mapping[str, MarketEvent | None]


class StockDataStream(Protocol):
    """Structural contract for a replayable single-instrument loader."""

    def stream(self) -> Iterator[MarketEvent]: ...


class BacktestDataStream(Protocol):
    """Driver input: either raw single-stock events or synchronized batches."""

    def stream(self) -> Iterator[MarketEvent | MarketContext]: ...


class Side(Enum):
    BUY = 1
    SELL = 2


@dataclass
class Order:
    """A trade request from a strategy.

    BUY requests the full quantity, subject to execution-time cash and fees.
    SELL is clamped to execution-time holdings; short selling is not supported.
    reason is optional strategy-provided context for the request.
    """
    symbol: str
    side: Side
    qty: int = 1
    reason: str = ""


@dataclass(frozen=True)
class RejectedOrder:
    """An unfilled request; no cash, position, fees, or trades are changed."""
    symbol: str
    side: Side
    requested_qty: int
    execution_time: Timestamp
    execution_price: float
    reason: str
    order_reason: str = ""


@dataclass
class Trade:
    """A realized round-trip (or partial close) recorded when |position| shrinks.

    direction: +1 if the closed position was long, -1 if it was short.
    costs: allocated entry + exit trading costs (commissions / levies).
    financing: allocated short-borrow carry accrued while the closed
        quantity was held (0 for longs and when borrow_rate is 0).
    pnl: realized profit/loss net of `costs` AND `financing`.
    """
    symbol: str
    direction: int
    qty: int
    entry_price: float
    exit_price: float
    entry_time: Timestamp
    exit_time: Timestamp
    costs: float
    pnl: float
    financing: float = 0.0
    # Excursions while the closed quantity was open, gross currency on that qty:
    #   mfe >= 0  best unrealized gain reached before the close
    #   mae <= 0  worst unrealized loss reached before the close
    # Uses held candle highs/lows or observed tick prices, plus fill prices.
    # Continuous-position extremes vs average entry; not per-lot after scale-ins.
    # Diagnostic, not realized — used to tune stops/exits (see broker tracking).
    mfe: float = 0.0
    mae: float = 0.0
    # Reason on the order that realized this trade (including partial closes).
    exit_reason: str = ""

    @property
    def return_pct(self) -> float:
        """Net PnL as a fraction of the entry notional (0.02 == +2%).

        Normalizes PnL across position sizes so trades compare fairly. 0 when
        the entry notional is degenerate (zero price or qty)."""
        notional = self.entry_price * self.qty
        return self.pnl / notional if notional else 0.0

    @property
    def duration(self) -> pd.Timedelta:
        """How long the closed quantity was held (exit_time - entry_time).

        Works for both sources: tick times are int epoch-ns, candle times are
        pd.Timestamp; pd.to_datetime normalizes both before subtracting."""
        return pd.to_datetime(self.exit_time) - pd.to_datetime(self.entry_time)


@dataclass(frozen=True)
class BuyAndHoldResult:
    """Independent gross baseline: full capital, fractional shares, no fees.

    Uses the actual first and last observed closes, without skipping invalid
    endpoints. Revenue and return are NaN when an endpoint is not positive
    and finite. Dates can differ between symbols; empty sources are omitted.
    """
    symbol: str
    initial_capital: float
    entry_price: float
    exit_price: float
    entry_time: Timestamp
    exit_time: Timestamp
    revenue: float
    return_pct: float


@dataclass
class BacktestResult:
    revenue: float            # final_equity - initial_capital
    return_pct: float         # revenue / initial_capital * 100
    sharpe: float             # daily returns, annualized x sqrt(252)
    max_drawdown: float       # positive fraction, e.g. 0.12 == 12%
    n_trades: int
    win_rate: float           # winners / n_trades (0..1)
    avg_win: float
    avg_loss: float
    equity_curve: pd.Series   # timestamp-indexed equity
    trades: List[Trade] = field(default_factory=list)
    buy_and_hold: dict[str, BuyAndHoldResult] = field(default_factory=dict)
    rejected_orders: List[RejectedOrder] = field(default_factory=list)

    @property
    def avg_calendar_days(self) -> float:
        """Mean elapsed calendar days per closed trade, including partial closes.

        Includes weekends, holidays, and fractional days. Each trade has equal
        weight; returns 0.0 when there are no closed trades.
        """
        if not self.trades:
            return 0.0
        return sum(t.duration / pd.Timedelta(days=1) for t in self.trades) / len(self.trades)

    @property
    def return_distribution(self) -> str:
        """Printable closed-trade return histogram with 10 automatic bins."""
        # metrics imports BacktestResult; defer the reverse import to avoid a cycle.
        from kiseki.metrics import format_return_distribution

        return format_return_distribution(self.trades)

    def trades_df(self) -> pd.DataFrame:
        """The trade log as a DataFrame, one row per closed round-trip.

        Includes the derived `return_pct` and `duration` so refinement is a
        one-liner, e.g.:
            df.sort_values("pnl").head(10)               # 10 worst trades
            df[df.return_pct < -0.02]                    # trades past a threshold
        Returns an empty frame (with the expected columns) when there are no
        trades, so downstream code can rely on the schema."""
        columns = [
            "symbol", "direction", "qty", "entry_price", "exit_price",
            "entry_time", "exit_time", "duration",
            "costs", "financing", "pnl", "return_pct", "mfe", "mae", "exit_reason",
        ]
        rows = [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": pd.to_datetime(t.entry_time),
                "exit_time": pd.to_datetime(t.exit_time),
                "duration": t.duration,
                "costs": t.costs,
                "financing": t.financing,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
                "mfe": t.mfe,
                "mae": t.mae,
                "exit_reason": t.exit_reason,
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows, columns=columns)

    def summary(self) -> str:
        rows = [
            ("Revenue", f"{self.revenue:>15,.2f}"),
            ("Return", f"{self.return_pct:>14.2f}%"),
            ("Sharpe", f"{self.sharpe:>15.2f}"),
            ("Max drawdown", f"{self.max_drawdown * 100:>14.2f}%"),
            ("Trades", f"{self.n_trades:>15d}"),
            ("Rejected", f"{len(self.rejected_orders):>15d}"),
            ("Win rate", f"{self.win_rate * 100:>14.1f}%"),
            ("Avg win", f"{self.avg_win:>15,.2f}"),
            ("Avg loss", f"{self.avg_loss:>15,.2f}"),
            ("Avg cal days", f"{self.avg_calendar_days:>15.2f}"),
        ]
        width = 34
        line = "+" + "-" * (width - 2) + "+"
        out = [line, "|" + "Backtest Result".center(width - 2) + "|", line]
        for label, value in rows:
            out.append(f"| {label:<12}{value:>{width - 16}} |")
        out.append(line)
        if self.buy_and_hold:
            out.extend(["", "Buy & Hold Baselines (gross; full initial capital per symbol)",
                        "First/last observed close; fractional shares; no fees."])
            headers = ("Symbol", "Start", "End", "Revenue", "Return")
            baseline_rows = [
                (baseline.symbol,
                 str(pd.to_datetime(baseline.entry_time).date()),
                 str(pd.to_datetime(baseline.exit_time).date()),
                 f"{baseline.revenue:,.2f}", f"{baseline.return_pct:.2f}%")
                for _, baseline in sorted(self.buy_and_hold.items())
            ]
            widths = [max(len(row[i]) for row in [headers, *baseline_rows])
                      for i in range(len(headers))]
            border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

            def format_row(row):
                cells = [value.ljust(widths[i]) if i < 3 else value.rjust(widths[i])
                         for i, value in enumerate(row)]
                return "| " + " | ".join(cells) + " |"

            out.extend([border, format_row(headers), border])
            out.extend(format_row(row) for row in baseline_rows)
            out.append(border)
        return "\n".join(out)
