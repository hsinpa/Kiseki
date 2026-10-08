"""Batch and rolling median/IQR scaling for a single feature."""

from collections import deque
from numbers import Real

import numpy as np


class RobustScaler:
    """Scale values as (value - median) / (Q3 - Q1).

    Fit once on training values, then reuse for subsequent values of the same
    feature. Quartiles use linear interpolation. A zero or near-zero IQR uses
    a scale of 1, so constant training values become zero without division by
    zero. Outliers are not clipped: robust statistics do not bound the output.

    Alternatively, add observations one at a time with add(). Rolling statistics
    include the current observation and at most window_size - 1 earlier values.
    """

    def __init__(self, window_size: int = 60) -> None:
        if (isinstance(window_size, (bool, np.bool_))
                or not isinstance(window_size, (int, np.integer))
                or window_size < 1):
            raise ValueError("window_size must be a positive integer")
        self._values: deque[float] = deque(maxlen=int(window_size))
        self.center_: float | None = None
        self.scale_: float | None = None

    @staticmethod
    def _validate(values: list[float]) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.ndim != 1 or data.size == 0:
            raise ValueError("values must be a nonempty one-dimensional list of floats")
        if not np.isfinite(data).all():
            raise ValueError("values must contain only finite numbers")
        return data

    def fit(self, values: list[float]) -> "RobustScaler":
        """Fit all supplied values and seed rolling history with the latest ones.

        Batch fitting uses the entire list, regardless of window_size. A later
        add() switches to statistics from the bounded rolling history.
        """
        data = self._validate(values)
        center, scale = self._statistics(data)
        self._values.clear()
        self._values.extend(data[-self._values.maxlen:].tolist())
        self.center_, self.scale_ = center, scale
        return self

    @staticmethod
    def _statistics(data: np.ndarray) -> tuple[float, float]:
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        scale = float(q3 - q1)
        if not np.isfinite(scale):
            raise ValueError("values produce a nonfinite interquartile range")
        return float(median), scale if scale > 10 * np.finfo(float).eps else 1.0

    def add(self, value: float) -> float:
        """Add one finite observation, refit the rolling window, and scale it.

        Uses available observations during warm-up; the first result is 0.0.
        Only current/past observations are used. Previously returned values
        are not rescaled. Each call takes O(window_size) time and temporary
        memory. Invalid input leaves the fitted statistics/history unchanged.
        """
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("value must be a finite scalar number")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("value must be a finite scalar number")
        values = [*self._values, value][-self._values.maxlen:]
        center, scale = self._statistics(np.asarray(values, dtype=float))
        result = (value - center) / scale
        if not np.isfinite(result):
            raise ValueError("scaling produced nonfinite values")
        self._values.append(value)
        self.center_, self.scale_ = center, scale
        return result

    def transform(self, values: list[float]) -> list[float]:
        """Scale values with fitted statistics, without refitting or mutation."""
        if self.center_ is None or self.scale_ is None:
            raise ValueError("call fit before transform")
        data = self._validate(values)
        result = (data - self.center_) / self.scale_
        if not np.isfinite(result).all():
            raise ValueError("scaling produced nonfinite values")
        return result.tolist()

    def fit_transform(self, values: list[float]) -> list[float]:
        """Learn statistics from values and return their scaled values."""
        return self.fit(values).transform(values)
