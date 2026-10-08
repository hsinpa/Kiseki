import math
from bisect import bisect_right
from typing import List

import pandas as pd

from kiseki.types import Trade, BacktestResult, Timestamp

TRADING_DAYS_PER_YEAR = 252


def format_return_distribution(
    trades: List[Trade], bins: int = 10, height: int = 10
) -> str:
    """Format a vertical histogram of closed-trade net returns in percent.

    Bins span the observed range; intervals are [lower, upper), except the
    final interval includes its upper edge. Bar heights are rounded up to
    fit at most `height` rows. The legend reports exact counts and ranges.
    Non-finite returns are excluded and reported, never silently counted.
    """
    for name, value in (("bins", bins), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    title = "Closed-Trade Return Distribution"
    if not trades:
        return f"{title}\nNo closed trades available."

    returns = [trade.return_pct * 100 for trade in trades]
    finite_returns = [value for value in returns if math.isfinite(value)]
    excluded = len(returns) - len(finite_returns)
    exclusion_note = f"Excluded non-finite returns: {excluded}"
    if not finite_returns:
        return f"{title}\nNo finite trade returns available.\n{exclusion_note}"

    lower, upper = min(finite_returns), max(finite_returns)
    if lower == upper:
        # A two-percentage-point span keeps single/constant samples readable.
        lower, upper = lower - 1.0, upper + 1.0
        if lower == upper:
            previous = math.nextafter(lower, -math.inf)
            following = math.nextafter(upper, math.inf)
            lower = previous if math.isfinite(previous) else lower
            upper = following if math.isfinite(following) else upper
    # Weighted interpolation avoids overflow in (upper - lower).
    edges = [lower * (1 - i / bins) + upper * (i / bins)
             for i in range(bins + 1)]
    edges[0], edges[-1] = lower, upper
    counts = [0] * bins
    for value in finite_returns:
        index = min(bins - 1, max(0, bisect_right(edges, value) - 1))
        counts[index] += 1

    centers = [edges[i] / 2 + edges[i + 1] / 2 for i in range(bins)]

    def percentage_labels(values: List[float]) -> List[str]:
        for precision in range(2, 18):
            labels = [f"{value:+.{precision}f}%" for value in values]
            if len(set(labels)) == len(set(values)):
                return labels
        return [f"{value:+.17g}%" for value in values]

    labels = percentage_labels(centers)
    edge_labels = percentage_labels(edges)
    column_width = max(7, max(map(len, labels)) + 2)
    peak = max(counts)
    rows = min(height, peak)
    bar_heights = [(count * rows + peak - 1) // peak for count in counts]
    axis_width = len(str(peak))
    lines = [title, f"Closed trades: {len(finite_returns)}", "Number of trades"]
    for row in range(rows, 0, -1):
        tick = (row * peak + rows - 1) // rows
        bars = "".join(("#" if bar >= row else "").center(column_width)
                       for bar in bar_heights)
        lines.append(f"{tick:>{axis_width}} |{bars}".rstrip())
    lines.append(f"{0:>{axis_width}} +" + "-" * (column_width * bins))
    indent = " " * (axis_width + 2)
    lines.append(indent + "".join(label.center(column_width) for label in labels))
    lines.append(indent + "Trade loss/gain (%) - bin centers")
    if peak > rows:
        lines.append("Bar heights rounded up; exact counts below.")
    lines.extend(["", "Return range (%) | Number of trades"])
    ranges = [f"[{edge_labels[i]}, {edge_labels[i + 1]}"
              + ("]" if i == bins - 1 else ")") for i in range(bins)]
    range_width = max(map(len, ranges))
    lines.extend(f"{interval:<{range_width}} | {count}"
                 for interval, count in zip(ranges, counts))
    if excluded:
        lines.append(exclusion_note)
    return "\n".join(lines)


def _sharpe(equity_curve: pd.Series, initial_capital: float) -> float:
    """Daily-resampled, annualized Sharpe (risk-free 0).

    The first observed day's return is measured against initial capital, even
    though resampling keeps only that day's last equity observation. No prior
    calendar day is fabricated. Returns nan with fewer than two valid returns
    or zero return variance; returns from zero equity are undefined.
    """
    if equity_curve.empty:
        return math.nan
    daily = equity_curve.resample("1D").last().dropna()
    if daily.empty:
        return math.nan
    previous = daily.shift(1)
    previous.iloc[0] = initial_capital
    returns = (daily / previous.replace(0.0, math.nan) - 1).dropna()
    if len(returns) < 2:
        return math.nan
    std = returns.std()
    if std == 0 or math.isnan(std):
        return math.nan
    return returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.12 == 12%)."""
    if equity_curve.empty:
        return 0.0
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    worst = drawdown.min()
    return abs(worst) if worst < 0 else 0.0


def build_result(
    timestamps: List[Timestamp],
    equity: List[float],
    trades: List[Trade],
    initial_capital: float,
) -> BacktestResult:
    """Assemble results from post-batch equity observations and closed trades.

    For nonempty runs, prepend initial capital at the first event's timestamp.
    The duplicate timestamp represents pre-trade then post-batch equity, not
    an extra event or trading day. Empty runs retain an empty equity curve.
    Input lists are not modified.
    """
    index = pd.to_datetime(pd.Index(timestamps))
    equity_curve = pd.Series(equity, index=index, dtype="float64")
    if not equity_curve.empty:
        equity_curve = pd.Series(
            [initial_capital, *equity], index=index.insert(0, index[0]),
            dtype="float64",
        )

    final_equity = equity[-1] if equity else initial_capital
    revenue = final_equity - initial_capital
    return_pct = (revenue / initial_capital * 100) if initial_capital else 0.0

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n_trades = len(trades)

    return BacktestResult(
        revenue=revenue,
        return_pct=return_pct,
        sharpe=_sharpe(equity_curve, initial_capital),
        max_drawdown=_max_drawdown(equity_curve),
        n_trades=n_trades,
        win_rate=(len(wins) / n_trades) if n_trades else 0.0,
        avg_win=(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss=(sum(losses) / len(losses)) if losses else 0.0,
        equity_curve=equity_curve,
        trades=trades,
    )
