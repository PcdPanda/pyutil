import datetime as dt
from typing import Union

import pandas as pd


DatetimeLike = Union[dt.datetime, str, int]
DateLike = Union[dt.date, str]
FreqLike = Union[pd.PeriodDtype, str]
TimeLike = Union[dt.time, str]
