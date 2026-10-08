"""Market-price checks shared by ingestion, batch execution, and brokers.

Bad data fails loudly; prices are never repaired or silently skipped. Locked
quotes are valid, and a last trade need not lie inside the current bid/ask.
"""

import math
from numbers import Real

import numpy as np
import pandas as pd

from kiseki.utility.tick_aggregator import Candlestick, Tick


def validate_market_event(event, context: str = "Market event") -> None:
    """Require positive finite prices, ordered quotes, and consistent OHLC."""
    if isinstance(event, Tick):
        fields = ("close", "bid_price", "ask_price")
        when = event.ts
    elif isinstance(event, Candlestick):
        fields = ("open", "high", "low", "close")
        when = event.date
    else:
        raise TypeError(f"{context}: expected Tick or Candlestick event")
    context = f"{context} at {when}"
    for field in fields:
        value = getattr(event, field)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{context}: {field} must be finite and positive; got {value!r}")
    if isinstance(event, Tick):
        if event.bid_price > event.ask_price:
            raise ValueError(f"{context}: bid_price must be <= ask_price")
    elif (event.low > event.high
          or event.high < max(event.open, event.close)
          or event.low > min(event.open, event.close)):
        raise ValueError(f"{context}: inconsistent OHLC; low <= open/close <= high required")


def validate_market_frame(
    frame: pd.DataFrame, *, candle: bool = False, context: str = "Market data"
) -> None:
    """Vectorized checks for CSV prices and in-memory tick aggregation.

    Tick columns are lowercase; candle columns use the CSV's title-case schema.
    Checks operate on temporary arrays without modifying the input DataFrame.
    """
    fields = ("Open", "High", "Low", "Close") if candle else ("close", "bid_price", "ask_price")
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing price columns: {', '.join(missing)}")
    if not frame.columns.is_unique:
        raise ValueError(f"{context}: column names must be unique")
    values = {}
    for field in fields:
        if pd.api.types.is_bool_dtype(frame[field].dtype):
            raise ValueError(f"{context}: {field} must contain numeric prices, not booleans")
        try:
            prices = frame[field].to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context}: {field} must contain numeric prices") from exc
        invalid = ~np.isfinite(prices) | (prices <= 0)
        if invalid.any():
            row = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"{context}, data row {row + 1} (index {frame.index[row]!r}): "
                f"{field} must be finite and positive; got {prices[row]!r}"
            )
        values[field] = prices
    if candle:
        invalid = ((values["Low"] > values["High"])
                   | (values["High"] < np.maximum(values["Open"], values["Close"]))
                   | (values["Low"] > np.minimum(values["Open"], values["Close"])))
        rule = "inconsistent OHLC; low <= open/close <= high required"
    else:
        invalid = values["bid_price"] > values["ask_price"]
        rule = "bid_price must be <= ask_price"
    if invalid.any():
        row = int(np.flatnonzero(invalid)[0])
        raise ValueError(f"{context}, data row {row + 1} (index {frame.index[row]!r}): {rule}")
