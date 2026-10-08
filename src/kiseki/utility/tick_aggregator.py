from dataclasses import dataclass
from typing import Optional, List, Protocol, runtime_checkable
from collections import deque
import pandas as pd

@runtime_checkable
class HasOHLCV(Protocol):
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class Candlestick:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class OHLCVBar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    buy_vol: int          # tick_type == 1 (外盤)
    sell_vol: int         # tick_type == 2 (內盤)
    neutral_vol: int      # tick_type == 0
    vwap: float
    tick_count: int
    # Quote snapshot at last tick of bar
    bid_price: float
    ask_price: float
    bid_volume: int
    ask_volume: int

    @staticmethod
    def volume_imbalance(buy_vol: int, sell_vol: int) -> Optional[float]:
        """Signed trade-volume imbalance, excluding neutral volume.

        Return None when there is no positive classified volume.
        """
        total = buy_vol + sell_vol
        if total <= 0:
            return None
        return (buy_vol - sell_vol) / total

    @staticmethod
    def upper_wick_ratio(
        open: float, high: float, low: float, close: float
    ) -> Optional[float]:
        """Upper wick as a fraction of the bar's high-low range.

        Return None when the bar has no positive price range.
        """
        price_range = high - low
        if price_range <= 0:
            return None
        return (high - max(open, close)) / price_range

    # ---- Derived microstructure features (point-in-time, 'last' snapshot) ----
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid(self) -> float:
        return (self.ask_price + self.bid_price) / 2.0

    @property
    def rel_spread(self) -> Optional[float]:
        m = self.mid
        return self.spread / m if m > 0 else None

    @property
    def microprice(self) -> Optional[float]:
        total = self.bid_volume + self.ask_volume
        if total <= 0:
            return None
        # Note the cross-weighting: bid_price weighted by ASK volume, and vice versa.
        # Intuition: heavy ask-side liquidity pulls "fair value" toward the bid.
        return (self.bid_price * self.ask_volume +
                self.ask_price * self.bid_volume) / total

    @property
    def obi(self) -> Optional[float]:
        total = self.bid_volume + self.ask_volume
        if total <= 0:
            return None
        return (self.bid_volume - self.ask_volume) / total


@dataclass
class FeatureBar(OHLCVBar):
    """ Custom index for analysis """
    atr: float
    rvol: float
    session_vwap: float

@dataclass
class Tick:
    ts: int
    close: float
    volume: int
    bid_price: float
    bid_volume: int
    ask_price: float
    ask_volume: int
    tick_type: int        # 0 neutral, 1 buy, 2 sell

class TickAggregator:
    """Aggregates ticks into OHLCV+quote bars for ONE timeframe.

    period: pandas offset alias, e.g. '1min', '5min', '1h', '1D', '1W', '1MS'.
    """

    _CALENDAR_PERIODS = {
        "1W": "W-SUN",
        "1MS": "M",
    }

    def __init__(self, period: str, max_bars: int = 90):
        self.period = period
        self.max_bars = max_bars
        self._calendar_frequency = self._CALENDAR_PERIODS.get(period)
        self._bucket_ns = (
            None if self._calendar_frequency else pd.Timedelta(period).value
        )
        self._calendar_bucket_end_ns: Optional[int] = None

        self._bars: deque[OHLCVBar] = deque(maxlen=max_bars)
        self._current_bucket_ns: Optional[int] = None
        self._cur: Optional[OHLCVBar] = None
        self._pv_sum: float = 0.0
        self.process_count = 0

    # ---------- ingestion ----------

    def update(self, tick: Tick) -> Optional[OHLCVBar]:
        bucket_ns = self._bucket_start_ns(tick.ts)

        if bucket_ns == self._current_bucket_ns:
            self._merge_tick(tick)
            return None

        if self._current_bucket_ns is None:
            self._open_bar(bucket_ns, tick)
            return None

        if bucket_ns > self._current_bucket_ns:
            closed = self._close_bar()
            self._open_bar(bucket_ns, tick)
            self.process_count += 1
            return closed

        return None

    def _bucket_start_ns(self, timestamp_ns: int) -> int:
        if self._bucket_ns is not None:
            return timestamp_ns - (timestamp_ns % self._bucket_ns)

        if (
            self._current_bucket_ns is not None
            and self._calendar_bucket_end_ns is not None
            and self._current_bucket_ns <= timestamp_ns < self._calendar_bucket_end_ns
        ):
            return self._current_bucket_ns

        period = pd.Timestamp(timestamp_ns).to_period(self._calendar_frequency)
        self._calendar_bucket_end_ns = (period + 1).start_time.value
        return period.start_time.value

    def _open_bar(self, bucket_ns: int, tick: Tick) -> None:
        p, v, tt = tick.close, tick.volume, tick.tick_type
        bucket: pd.Timestamp = pd.Timestamp(bucket_ns)   # naive, interpreted as local

        self._current_bucket_ns = bucket_ns
        self._current_bucket = bucket

        self._cur = OHLCVBar(
            date=bucket,
            open=p, high=p, low=p, close=p,
            volume=v,
            buy_vol=v if tt == 1 else 0,
            sell_vol=v if tt == 2 else 0,
            neutral_vol=v if tt == 0 else 0,
            vwap=p,
            tick_count=1,
            bid_price=tick.bid_price,
            ask_price=tick.ask_price,
            bid_volume=tick.bid_volume,
            ask_volume=tick.ask_volume,
        )
        self._pv_sum = p * v

    def _merge_tick(self, tick: Tick) -> None:
        bar = self._cur
        p, v, tt = tick.close, tick.volume, tick.tick_type
        if p > bar.high: bar.high = p
        if p < bar.low:  bar.low = p
        bar.close = p
        bar.volume += v
        if tt == 1:   bar.buy_vol += v
        elif tt == 2: bar.sell_vol += v
        else:         bar.neutral_vol += v
        self._pv_sum += p * v
        bar.vwap = self._pv_sum / bar.volume if bar.volume > 0 else p
        bar.tick_count += 1

        # Quote = 'last': overwrite on every tick
        bar.bid_price  = tick.bid_price
        bar.ask_price  = tick.ask_price
        bar.bid_volume = tick.bid_volume
        bar.ask_volume = tick.ask_volume

    def _close_bar(self) -> Optional[OHLCVBar]:
        if self._cur is None:
            return None
        finalized = self._cur
        self._bars.append(finalized)
        self._cur = None
        self._pv_sum = 0.0
        return finalized

    # ---------- query ----------

    def get_full(self, n: Optional[int] = None) -> List[OHLCVBar]:
        return list(self._bars) if n is None else list(self._bars)[-n:]

    def get_partial(self) -> Optional[OHLCVBar]:
        return self._cur

    def get_all(self, n: Optional[int] = None) -> List[OHLCVBar]:
        bars = list(self._bars)
        if self._cur is not None:
            bars.append(self._cur)
        return bars if n is None else bars[-n:]

    def latest_full(self) -> Optional[OHLCVBar]:
        return self._bars[-1] if self._bars else None

    @property
    def count(self) -> int:
        return len(self._bars)


