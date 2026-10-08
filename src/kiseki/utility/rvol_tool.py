from collections import deque


class RollingRVOL:
    """
    Rolling Relative Volume calculator with O(1) lookup.
    """
    def __init__(self, window: int = 20):
        self.window = window
        self._volumes: deque[float] = deque(maxlen=window)
        self._vol_sum = 0.0
        self._cache_values = {}

    def add(self, volume: float) -> None:
        """Add a bar's volume."""
        if len(self._volumes) >= self.window:
            self._vol_sum -= self._volumes[0]

        self._volumes.append(volume)
        self._vol_sum += volume

    def get(self) -> float:
        """
        Get current RVOL.

        Returns:
            Current volume / Average of previous bars
            0.0 if insufficient data
        """
        n = len(self._volumes)
        if n < 2:
            return 0.0

        current = self._volumes[-1]
        prev_sum = self._vol_sum - current

        if prev_sum == 0:
            return 0.0

        prev_avg = prev_sum / (n - 1)
        return current / prev_avg

    def get_avg_volume(self) -> float:
        """Get average volume of all bars in window."""
        if len(self._volumes) == 0:
            return 0.0
        return self._vol_sum / len(self._volumes)

    def reset(self) -> None:
        """Clear all data."""
        self._volumes.clear()
        self._vol_sum = 0.0

    def __len__(self) -> int:
        return len(self._volumes)