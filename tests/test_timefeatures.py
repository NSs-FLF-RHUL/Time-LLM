import numpy as np
from pandas import DatetimeIndex

import timellm.utils.timefeatures as tf


def test_second_of_minute_call() -> None:
    """Test second of minute encoding."""
    t = DatetimeIndex(["01/01/2026 12:30:00+00:00", "01/01/2026 12:30:59+00:00"])
    sof = tf.SecondOfMinute()
    np.testing.assert_array_equal(sof(t), np.array([-0.5, 0.5]))


def test_second_of_minute_reverse() -> None:
    """Test second of minute decoding."""
    t = np.array([-0.5, 0.5])
    sof = tf.SecondOfMinute()
    np.testing.assert_array_equal(sof.reverse(t), np.array([0.0, 59.0]))


def test_minute_of_hour_call() -> None:
    """Test minute of hour encoding."""
    t = DatetimeIndex(["01/01/2026 12:00:00+00:00", "01/01/2026 12:59:59+00:00"])
    moh = tf.MinuteOfHour()
    np.testing.assert_array_equal(moh(t), np.array([-0.5, 0.5]))


def test_minute_of_hour_reverse() -> None:
    """Test minute of hour decoding."""
    t = np.array([-0.5, 0.5])
    moh = tf.MinuteOfHour()
    np.testing.assert_array_equal(moh.reverse(t), np.array([0.0, 59.0]))


def test_hour_of_day_call() -> None:
    """Test hour of day encoding."""
    t = DatetimeIndex(["01/01/2026 00:00:00+00:00", "01/01/2026 23:59:59+00:00"])
    mod = tf.HourOfDay()
    np.testing.assert_array_equal(mod(t), np.array([-0.5, 0.5]))


def test_hour_of_day_reverse() -> None:
    """Test hour of day decoding."""
    t = np.array([-0.5, 0.5])
    mod = tf.HourOfDay()
    np.testing.assert_array_equal(mod.reverse(t), np.array([0.0, 23.0]))


def test_day_of_week_call() -> None:
    """Test day of week encoding."""
    t = DatetimeIndex(
        [
            "2026/10/12 09:00:00+00:00",
            "19/10/2026 09:00:00+00:00",
            "18/10/2026 09:00:00+00:00",
        ]
    )
    dow = tf.DayOfWeek()
    np.testing.assert_array_equal(dow(t), np.array([-0.5, -0.5, 0.5]))


def test_day_of_month_call() -> None:
    """Test day of month encoding."""
    t = DatetimeIndex(["01/01/2026 00:00:00+00:00", "31/01/2026 00:00:00+00:00"])
    dom = tf.DayOfMonth()
    np.testing.assert_array_equal(dom(t), np.array([-0.5, 0.5]))


def test_day_of_year_call() -> None:
    """Test day of year encoding."""
    t = DatetimeIndex(
        [
            "01/01/2026 00:00:00+00:00",
            "2026/12/31 23:00:00+00:00",
            "2020/12/31 23:00:00+00:00",  # leap year
        ]
    )
    doy = tf.DayOfYear()
    np.testing.assert_array_equal(doy(t), np.array([-0.5, 0.49726027397260275, 0.5]))


def test_day_of_year_reverse() -> None:
    """Test day of year decoding."""
    t = np.array([-0.5, 0.5])
    doy = tf.DayOfYear()
    np.testing.assert_array_equal(doy.reverse(t), np.array([1.0, 366]))


def test_month_of_year_call() -> None:
    """Test month of year encoding."""
    t = DatetimeIndex(["2026/01/01 00:00:00+00:00", "2026/12/01 00:00:00+00:00"])
    moy = tf.MonthOfYear()
    np.testing.assert_array_equal(moy(t), np.array([-0.5, 0.5]))


def test_week_of_year_call() -> None:
    """Test week of year decoding."""
    t = DatetimeIndex(["2026/01/01 00:00:00+00:00", "2026/12/31 00:00:00+00:00"])
    woy = tf.WeekOfYear()
    np.testing.assert_array_equal(woy(t), np.array([-0.5, 0.5]))
