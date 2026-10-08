from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, List

import numpy as np
import pandas as pd

from kiseki.utility.tick_aggregator import Tick, Candlestick, OHLCVBar
from kiseki.utility.tick_utility import read_csv
from kiseki.types import MarketContext, MarketEvent, StockDataStream
from kiseki.validation import validate_market_event, validate_market_frame


@dataclass(frozen=True)
class StockSource:
    """A named single-instrument input; loader must provide stream()."""

    symbol: str
    loader: StockDataStream


class MultiStockDataLoader:
    """Merge all tradable streams into equal-time batches.

    Candles must be completion-labeled and have strictly increasing times per
    symbol. Tick duplicates preserve source order in separate rounds. Missing
    observations are None; exhausted symbols retain their last snapshot.
    Naive candle times must share a time basis; aware/naive mixing is rejected.
    """

    def __init__(self, sources: list[StockSource]):
        if not sources:
            raise ValueError("sources must not be empty")
        symbols = set()
        for source in sources:
            if not isinstance(source, StockSource):
                raise TypeError("sources must contain StockSource objects")
            if not isinstance(source.symbol, str) or not source.symbol.strip():
                raise ValueError("source symbols must be nonempty strings")
            if source.symbol in symbols:
                raise ValueError(f"Duplicate source symbol: {source.symbol}")
            if not callable(getattr(source.loader, "stream", None)):
                raise TypeError(f"Source {source.symbol}: loader must provide stream()")
            symbols.add(source.symbol)
        self.sources = tuple(sorted(sources, key=lambda source: source.symbol))

    @staticmethod
    def _timestamp(event: MarketEvent, symbol: str,
                   candle_awareness: set[bool]) -> int:
        if isinstance(event, Tick):
            if not isinstance(event.ts, Integral) or isinstance(event.ts, bool):
                raise ValueError(f"Source {symbol}: tick ts must be integer nanoseconds")
            timestamp = int(event.ts)
        elif isinstance(event, Candlestick):
            when = event.date
            if not isinstance(when, pd.Timestamp) or pd.isna(when):
                raise ValueError(f"Source {symbol}: candle date must be a valid Timestamp")
            candle_awareness.add(when.tzinfo is not None)
            if len(candle_awareness) > 1:
                raise ValueError(f"Source {symbol}: mixed aware/naive candle timestamps")
            try:
                timestamp = when.as_unit("ns").value
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Source {symbol}: timestamp outside nanosecond range") from exc
        else:
            raise TypeError(f"Source {symbol}: expected Tick or Candlestick event")
        if not -(2**63) < timestamp <= 2**63 - 1:
            raise ValueError(f"Source {symbol}: timestamp outside nanosecond range")
        return timestamp

    def _checked_stream(self, source: StockSource,
                        candle_awareness: set[bool]) -> Iterator[tuple[int, MarketEvent]]:
        previous = None
        for event in source.loader.stream():
            timestamp = self._timestamp(event, source.symbol, candle_awareness)
            validate_market_event(event, f"Source {source.symbol}")
            if previous is not None and timestamp < previous:
                raise ValueError(f"Source {source.symbol}: timestamps must be nondecreasing")
            if previous == timestamp and isinstance(event, Candlestick):
                raise ValueError(f"Source {source.symbol}: candle timestamps must be unique")
            previous = timestamp
            yield timestamp, event

    def stream(self) -> Iterator[MarketContext]:
        """Yield sorted-symbol batches until every source is exhausted.

        State resets per replay. One future event per source remains private;
        coordinator memory is O(number of sources), excluding loader buffers.
        """
        candle_awareness: set[bool] = set()
        streams = {source.symbol: self._checked_stream(source, candle_awareness)
                   for source in self.sources}
        latest = dict.fromkeys(streams)
        lookahead = {symbol: next(iterator, None)
                     for symbol, iterator in streams.items()}
        while any(item is not None for item in lookahead.values()):
            timestamp = min(item[0] for item in lookahead.values() if item is not None)
            events = {symbol: item[1] for symbol, item in lookahead.items()
                      if item is not None and item[0] == timestamp}
            latest.update(events)
            yield MarketContext(MappingProxyType(events),
                                MappingProxyType(latest.copy()))
            for symbol in events:
                lookahead[symbol] = next(streams[symbol], None)


class BaseDataLoader:
    """Streams a time-ordered record stream from CSV files in a folder.

    Memory-bounded: files are processed one at a time in ascending filename
    order, and each file's data is released before the next is loaded. Records
    are sorted within each file only. This assumes files do not overlap in time
    and are named chronologically (the normal one-file-per-session dump layout).

    Subclasses provide `_parse(df)` (rows -> records) and `_sort_key(record)`.
    """

    #: Human-readable name of the records this loader produces (for errors).
    record_name = "record"

    def __init__(self, folder: str):
        self.folder = Path(folder)

    def _parse(self, df) -> List:
        raise NotImplementedError

    def _sort_key(self, record):
        raise NotImplementedError

    def _select_paths(self, paths: list[Path]) -> list[Path]:
        return paths

    def _discover_paths(self) -> list[Path]:
        if not self.folder.exists():
            raise FileNotFoundError(f"Data folder does not exist: {self.folder}")

        paths = sorted(self.folder.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(
                f"No CSV {self.record_name} files found in: {self.folder}"
            )

        return paths

    def stream(self) -> Iterator:
        """Yield records lazily, one CSV file at a time, in ascending time order."""
        paths = self._select_paths(self._discover_paths())
        for path in paths:
            # Load a single file; `df` goes out of scope after this iteration,
            # so only one file's rows are held in memory at a time.
            df = read_csv([str(path)])
            try:
                records = self._parse(df)
            except (ValueError, TypeError, OverflowError, AttributeError) as exc:
                raise ValueError(f"Invalid {self.record_name} data in {path}: {exc}") from exc
            records.sort(key=self._sort_key)
            yield from records


class TickDataLoader(BaseDataLoader):
    """Loads a Tick stream from CSV files with the Tick schema:
    ts, close, volume, bid_price, bid_volume, ask_price, ask_volume, tick_type.

    start_date and end_date are optional inclusive calendar-date bounds for
    files named YYYY-MM-DD.csv. A boundary date does not need to have a
    corresponding file; only available files inside the range are loaded.
    """

    record_name = "tick"

    def __init__(
        self,
        folder: str,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ):
        super().__init__(folder)
        self.start_date = self._validate_date("start_date", start_date)
        self.end_date = self._validate_date("end_date", end_date)
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date.date() > self.end_date.date()
        ):
            raise ValueError("start_date must be on or before end_date")

    @staticmethod
    def _validate_date(
        name: str, value: pd.Timestamp | None
    ) -> pd.Timestamp | None:
        if value is not None and not isinstance(value, pd.Timestamp):
            raise TypeError(f"{name} must be a pandas Timestamp or None")
        return value

    def _select_paths(self, paths: list[Path]) -> list[Path]:
        if self.start_date is None and self.end_date is None:
            return paths

        start = self.start_date.date() if self.start_date is not None else None
        end = self.end_date.date() if self.end_date is not None else None
        selected: list[Path] = []
        for path in paths:
            try:
                file_date = pd.Timestamp(path.stem).date()
            except ValueError:
                continue
            if start is not None and file_date < start:
                continue
            if end is not None and file_date > end:
                continue
            selected.append(path)
        return selected

    def _parse(self, df) -> List[Tick]:
        validate_market_frame(df, context="Tick CSV")
        return [
            Tick(
                ts=int(row.ts),
                close=float(row.close),
                volume=int(row.volume),
                bid_price=float(row.bid_price),
                bid_volume=int(row.bid_volume),
                ask_price=float(row.ask_price),
                ask_volume=int(row.ask_volume),
                tick_type=int(row.tick_type),
            )
            for row in df.itertuples(index=False)
        ]

    def _sort_key(self, record: Tick):
        return record.ts


class ParquetDataLoader:
    """Stream an already-loaded DataFrame as chronologically ordered ticks.

    Datetime timestamps are converted to integer nanoseconds; integer
    timestamps are already interpreted as nanoseconds. The caller's DataFrame
    is not modified. Ticks are constructed lazily, although normalization and
    sorting an unordered DataFrame require additional memory.
    """

    record_name = "tick"
    _required_columns = (
        "ts", "close", "volume", "bid_price", "bid_volume",
        "ask_price", "ask_volume", "tick_type",
    )

    def __init__(self, parquet_data: pd.DataFrame):
        if not isinstance(parquet_data, pd.DataFrame):
            raise TypeError("parquet_data must be a pandas DataFrame")
        missing = [
            column for column in self._required_columns
            if column not in parquet_data.columns
        ]
        if missing:
            raise ValueError(f"Missing required tick columns: {', '.join(missing)}")
        if not parquet_data.columns.is_unique:
            raise ValueError("parquet_data must have unique column names")
        self.parquet_data = parquet_data

    @staticmethod
    def _timestamps_ns(values: pd.Series) -> pd.Index:
        if values.isna().any():
            raise ValueError("ts must not contain missing timestamps")
        if pd.api.types.is_integer_dtype(values.dtype):
            # Reserve int64's minimum value for pandas' NaT sentinel.
            if values.min() <= -(2**63) or values.max() > 2**63 - 1:
                raise ValueError("ts contains timestamps outside the nanosecond range")
            return pd.Index(values.to_numpy(dtype="int64"))
        if (
            pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_timedelta64_dtype(values.dtype)
        ):
            raise ValueError("ts must contain datetimes or integer nanosecond timestamps")
        try:
            timestamps = pd.DatetimeIndex(pd.to_datetime(values)).as_unit("ns")
            if timestamps.hasnans:
                raise ValueError("missing timestamp")
            return pd.Index(timestamps.asi8)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ts contains invalid or out-of-range timestamps") from exc

    def _ordered_data(self) -> tuple[pd.DataFrame, pd.Index]:
        df = self.parquet_data
        validate_market_frame(df, context="Tick DataFrame")
        timestamps = self._timestamps_ns(df["ts"])
        if not timestamps.is_monotonic_increasing:
            order = timestamps.argsort(kind="stable")
            df = df.iloc[order]
            timestamps = timestamps.take(order)
        return df, timestamps

    def stream(self) -> Iterator[Tick]:
        """Yield ticks in ascending time order, preserving equal-time row order."""
        if self.parquet_data.empty:
            return
        df, timestamps = self._ordered_data()

        for timestamp, row in zip(timestamps, df.itertuples(index=False)):
            yield Tick(
                ts=int(timestamp),
                close=float(row.close),
                volume=int(row.volume),
                bid_price=float(row.bid_price),
                bid_volume=int(row.bid_volume),
                ask_price=float(row.ask_price),
                ask_volume=int(row.ask_volume),
                tick_type=int(row.tick_type),
            )

    def aggregate_bars(self, period: str) -> list[OHLCVBar]:
        """Batch-aggregate ticks without constructing a Python object per tick.

        NumPy reductions operate on contiguous time buckets. Like TickAggregator,
        fixed periods are epoch-aligned, weeks start Monday, months start on the
        first, empty buckets are omitted, and the final partial bar is included.
        Equal timestamps retain input order. Input data is not modified.

        This in-memory path uses additional tick-sized arrays and int64 volume
        sums; use stream() for incremental aggregation. Floating-point VWAP sums
        can differ from sequential accumulation by rounding error.
        """
        calendar_frequency = {"1W": "W-SUN", "1MS": "M"}.get(period)
        bucket_ns = None if calendar_frequency else pd.Timedelta(period).value
        if bucket_ns is not None and bucket_ns <= 0:
            raise ValueError("period must be positive")
        if self.parquet_data.empty:
            return []
        df, timestamps = self._ordered_data()
        timestamps = timestamps.to_numpy(dtype="int64")
        if calendar_frequency:
            buckets = (
                pd.to_datetime(timestamps).to_period(calendar_frequency)
                .start_time.as_unit("ns").asi8
            )
        else:
            buckets = timestamps - timestamps % bucket_ns

        starts = np.concatenate(([0], np.flatnonzero(buckets[1:] != buckets[:-1]) + 1))
        ends = np.concatenate((starts[1:], [len(df)])) - 1
        dates = pd.to_datetime(buckets[starts])
        del buckets

        prices = df["close"].to_numpy(dtype="float64")
        volumes = df["volume"].astype("int64", copy=False).to_numpy()
        tick_types = df["tick_type"].astype("int64", copy=False).to_numpy()
        first_prices = prices[starts]
        last_prices = prices[ends]
        # Source prices have already passed finite/positive validation.
        highs = np.maximum.reduceat(prices, starts)
        lows = np.minimum.reduceat(prices, starts)
        total_volume = np.add.reduceat(volumes, starts)
        price_volume = np.add.reduceat(prices * volumes, starts)
        counts = ends - starts + 1
        vwap = np.divide(
            price_volume, total_volume,
            out=last_prices.copy(), where=total_volume > 0,
        )
        # _open_bar sets VWAP to price even for a zero-volume single tick.
        vwap[counts == 1] = first_prices[counts == 1]

        buy_volume = np.add.reduceat(np.where(tick_types == 1, volumes, 0), starts)
        sell_volume = np.add.reduceat(np.where(tick_types == 2, volumes, 0), starts)
        neutral_volume = np.add.reduceat(
            np.where((tick_types != 1) & (tick_types != 2), volumes, 0), starts,
        )
        # Match the stream's treatment of nonstandard tick types: neutral on
        # merged ticks, but not on the opening tick unless its type is zero.
        opening_types = tick_types[starts]
        nonstandard = (opening_types != 0) & (opening_types != 1) & (opening_types != 2)
        neutral_volume -= np.where(nonstandard, volumes[starts], 0)

        aggregated = pd.DataFrame({
            "date": dates,
            "open": first_prices,
            "high": highs,
            "low": lows,
            "close": last_prices,
            "volume": total_volume,
            "buy_vol": buy_volume,
            "sell_vol": sell_volume,
            "neutral_vol": neutral_volume,
            "vwap": vwap,
            "tick_count": counts,
            "bid_price": df["bid_price"].to_numpy(dtype="float64")[ends],
            "ask_price": df["ask_price"].to_numpy(dtype="float64")[ends],
            "bid_volume": df["bid_volume"].astype("int64", copy=False).to_numpy()[ends],
            "ask_volume": df["ask_volume"].astype("int64", copy=False).to_numpy()[ends],
        })
        return [OHLCVBar(*row) for row in aggregated.itertuples(index=False, name=None)]


class CandleStickDataLoader(BaseDataLoader):
    """Loads a Candlestick stream from CSV files with the OHLCV schema:
    Date, Open, High, Low, Close, Volume.

    OHLC prices must be finite and positive, with low <= open/close <= high.

    Supply exactly one source: folder for a directory path, or csv_pathes for
    a nonempty list of CSV paths.
    Files are loaded in ascending path order, not list order; their time ranges
    must not overlap and their paths must sort chronologically.

    If timezone is supplied, interpret naive dates as UTC and convert dates to
    that timezone. Otherwise, preserve the original timestamp behavior.
    """

    record_name = "candlestick"

    def __init__(
        self,
        folder: str | Path | None = None,
        timezone: str | None = None,
        *,
        csv_pathes: list[str | Path] | None = None,
    ):
        if (folder is None) == (csv_pathes is None):
            raise ValueError("Supply exactly one of folder or csv_pathes")
        self.csv_paths = None
        if csv_pathes is not None:
            if not isinstance(csv_pathes, list):
                raise TypeError("csv_pathes must be a list of CSV paths")
            if not csv_pathes:
                raise ValueError("CSV paths must not be empty")
            self.csv_paths = [Path(path) for path in csv_pathes]
        else:
            if not isinstance(folder, (str, Path)):
                raise TypeError("folder must be a directory path; use csv_pathes for a list of CSV paths")
            super().__init__(folder)
        self.timezone = timezone

    def _discover_paths(self) -> list[Path]:
        if self.csv_paths is None:
            return super()._discover_paths()
        for path in self.csv_paths:
            if not path.is_file():
                raise FileNotFoundError(f"CSV file does not exist: {path}")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Expected a CSV file: {path}")
        return sorted(self.csv_paths)

    def _parse(self, df) -> List[Candlestick]:
        validate_market_frame(df, candle=True, context="Candle CSV")
        return [
            Candlestick(
                date=(
                    pd.Timestamp(row.Date)
                    if self.timezone is None
                    else pd.to_datetime(row.Date, utc=True).tz_convert(self.timezone)
                ),
                open=float(row.Open),
                high=float(row.High),
                low=float(row.Low),
                close=float(row.Close),
                volume=int(row.Volume),
            )
            for row in df.itertuples(index=False)
        ]

    def _sort_key(self, record: Candlestick):
        return record.date


# Backwards-compatible alias: the original tick-only loader.
DataLoader = TickDataLoader
