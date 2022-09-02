import pytest

from dateutil import tz
import datetime as dt
import pandas as pd

from pyutil import parse_to_time, parse_to_date, parse_to_datetime, parse_to_freq


def test_parse_to_datetime():
    dt_expected = dt.datetime(2020, 1, 1, 0, 0, 0)
    for ts in [dt_expected, pd.Timestamp("20200101"), dt_expected.timestamp()]:
        assert parse_to_datetime(ts) == dt_expected


def test_parse_to_datetime_err():
    for ts in [dt.date(2020, 1, 1), "20220101"]:
        with pytest.raises(TypeError):
            parse_to_datetime(ts)


def test_parse_to_date():
    date_expected = dt.date(2020, 1, 1)
    for date in [date_expected, "20200101", "2020/01/01"]:
        assert parse_to_date(date) == date_expected


def test_parse_to_date_err():
    for date in ["", "20200101 US/CT"]:
        with pytest.raises(ValueError):
            parse_to_date(date)

    for date in [dt.datetime.now(), pd.Timestamp.now(), 20200101]:
        with pytest.raises(TypeError):
            parse_to_date(date)


def test_parse_to_time():
    dt_expected = dt.time(12, 12, 12)
    for t in [dt_expected, "12:12:12", "121212", "12:12:12.000000"]:
        assert parse_to_time(t) == dt_expected

    assert parse_to_time("12:12:12.121212") == dt.time(12, 12, 12, 121212)

    date = dt.date(2020, 7, 1)
    dt_expected = dt.time(23, 23, 23, tzinfo=tz.gettz("Asia/Shanghai"))
    ts_expected = dt.datetime.combine(date, dt_expected).timestamp()
    for t in ["23:23:23 Asia/Shanghai", "15:23:23 UTC"]:
        assert dt.datetime.combine(date, parse_to_time(t)).timestamp() == ts_expected


def test_parse_to_time_err():
    for t in [121212]:
        with pytest.raises(TypeError):
            parse_to_time(t)

    for t in ["121212 China/Shanghai", "121212 US/CT "]:
        with pytest.raises(ValueError):
            parse_to_time(t)


def test_parse_to_freq():
    for freq_obj in ["7200S", "2H", pd.PeriodDtype("120Min")]:
        assert parse_to_freq(freq_obj) == "2h"
    for freq_obj in ["3Min", "180S", pd.PeriodDtype("3min")]:
        assert parse_to_freq(freq_obj) == "3min"

    with pytest.raises(TypeError):
        parse_to_freq(7200)
