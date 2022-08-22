import datetime as dt
from dateutil import tz
from ._typing import DateLike, DatetimeLike, TimeLike


def parse_to_datetime(dt_obj: DatetimeLike) -> dt.datetime:
    """
    A function to parse the given input to datetime.datetime

    Parameters
    ----------
        dt_obj: An object to be parsed.
            int will be treated as Unix Timestamp
            str will be parsed as "%H:%M:%S <tz>" or "%H%M%S <tz>"

    Returns
    -------
        datetime.datetime: The parsed datetime
    """
    if isinstance(dt_obj, dt.datetime):
        return dt_obj
    elif isinstance(dt_obj, (int, float)):
        return dt.datetime.fromtimestamp(dt_obj)
    else:
        raise TypeError(f"The input should be one of the following types:\n"
                        f"dt.datetime, pd.Timestamp, int, the given input is {type(dt_obj)}")


def parse_to_date(date_obj: DateLike) -> dt.date:
    """
    A function to parse the given input to datetime.date

    Parameters
    ----------
        date_obj (DateLike): An object to be parsed.
            str will be parsed as "%Y%m%d or %Y/%m/%d"

    Returns
    -------
        datetime.date: The parsed date
    """
    if isinstance(date_obj, dt.date) and not isinstance(date_obj, dt.datetime):
        return date_obj
    elif isinstance(date_obj, str):
        date_obj = date_obj.replace("/", "")
        date_obj = dt.datetime.strptime(date_obj, "%Y%m%d")
        return dt.date(date_obj.year, date_obj.month, date_obj.day)
    else:
        raise TypeError(f"The input should be one of the following types:\n"
                        f"dt.date, int, str, the given input is {type(date_obj)}")


def parse_to_time(time_obj: TimeLike) -> dt.time:
    """
    A function to parse the given input to datetime.time

    Parameters
    ==========
        time_obj (TimeLike): An object to be parsed.
            str will be parsed as "%H:%M:%S <tz>" or "%H%M%S <tz>"

    Returns
    -------
        datetime.time: The parsed time
    """
    if isinstance(time_obj, dt.time) and not isinstance(time_obj, dt.datetime):
        return time_obj
    elif isinstance(time_obj, str):
        time_obj = time_obj.replace(":", "")
        tzinfo = None
        if " " in time_obj:
            time_objs = time_obj.split(" ")
            if len(time_objs) != 2:
                raise ValueError(f"The given input with time zone should in the format "
                                 f"like %H:%M:%S <tz> or %H%M%S <tz>, but the given one "
                                 f"is {type(time_obj)}")
            tzinfo = tz.gettz(time_objs[1])
            if tzinfo is None:
                raise ValueError(f"The given tz {time_objs[1]} is invalid, it should be "
                                 f"parsed by dateutil.tz")
            time_obj = time_objs[0]
        time_obj = dt.datetime.strptime(time_obj, "%H%M%S")
        return dt.time(time_obj.hour, time_obj.minute, time_obj.second, tzinfo=tzinfo)
    else:
        raise TypeError(f"The input time should be one of the following types:\n"
                        f"dt.time, int, str, the given input is {type(time_obj)}")
