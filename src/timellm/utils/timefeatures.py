"""Time features for time series."""

import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """Time features for time series."""

    def __init__(self) -> None:
        """Initialize a TimeFeature object."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:  # noqa: D102
        pass

    def __repr__(self) -> str:  # noqa: D105
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode second of minute as floats."""
        return index.second / 59.0 - 0.5

    def reverse(self, t: np.ndarray) -> np.ndarray:
        """Decode floats as minutes."""
        return (t + 0.5) * 59.0


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode minute as floats."""
        return index.minute / 59.0 - 0.5

    def reverse(self, t: np.ndarray) -> np.ndarray:
        """Decode floats as minutes."""
        return (t + 0.5) * 59.0


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode hour of day as floats."""
        return index.hour / 23.0 - 0.5

    def reverse(self, t: np.ndarray) -> np.ndarray:
        """Decode floats as hour of day."""
        return (t + 0.5) * 23.0


class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode day of week as floats."""
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode day of month as float."""
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode day of year as floats."""
        return (index.dayofyear - 1) / 365.0 - 0.5

    def reverse(self, t: np.ndarray) -> np.ndarray:
        """Decode floats to day of year."""
        return (t + 0.5) * 365.0 + 1


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode month of year as floats."""
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Encode week of year as floats."""
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> list[TimeFeature]:
    """
    Return a list of appropriate time features for the given frequency string.

    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity]
        such as "12H", "5min", "1D" etc.

    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates: DatetimeIndex, freq: str = "h") -> np.ndarray:  # noqa: D103
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
