import datetime as dt
from pyutil import PyTensor

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def nested_dict():
    return {dt.datetime(2022, 1, 1, 0, 0): {"USDT": {"Cost": 1, "PNL": 0, "Pos": 15000,
                                                     "Realized PNL": 0, "Val": 15000, "Vol": 15000},
                                            "ETH": {"Cost": 0, "PNL": 0, "Pos": 0,
                                                    "Realized PNL": 0, "Val": 0, "Vol": 0}},
            dt.datetime(2022, 1, 1, 23, 59, 59): {"USDT": {"Cost": 1, "PNL": 0,
                                                           "Pos": 142, "Realized PNL": 0,
                                                           "Val": 142, "Vol": 142},
                                                  "ETH": {"Cost": 3718, "PNL": 189,
                                                          "Pos": 3.996, "Realized PNL": 0,
                                                          "Val": 15047, "Vol": 100}}}


@pytest.fixture
def slicing_dict():
    return {"small": {"first": [1, 3], "second": [2, 4]},
            "large": {"first": [10, 30], "second": [20, 40]}}


def test_construct_by_value():
    pt1 = PyTensor(1)
    pt2 = PyTensor(1.0)
    xt3 = PyTensor(np.NaN)
    assert pt1.dtype == int
    assert pt2.dtype == float
    assert xt3.dtype == float
    assert len(pt1.indexes) == 0
    assert len(pt1.shape) == 0


def test_construct_by_iterable():
    pt = PyTensor([1, 2.0, 3])
    assert pt.values.dtype == float
    assert pt.shape == (3,)
    assert (pt.values == np.array([1., 2., 3.])).all()


def test_construct_by_dict(nested_dict):
    dict_data = nested_dict[dt.datetime(2022, 1, 1, 23, 59, 59)]["ETH"]
    pt = PyTensor(dict_data)
    assert pt["Val"].values == dict_data["Val"]


def test_construct_by_series(nested_dict):
    series = pd.Series(nested_dict[dt.datetime(2022, 1, 1, 23, 59, 59)]["ETH"])
    pt = PyTensor(series)
    assert pt["Val"].values == series["Val"]


def test_construct_by_df(nested_dict):
    df = pd.DataFrame(nested_dict[dt.datetime(2022, 1, 1, 23, 59, 59)])
    pt = PyTensor(df)
    assert pt["Val", "ETH"].values == df["ETH"]["Val"]
    assert pt[4, 1].values == df["ETH"]["Val"]


def test_construct_by_nested_dict(nested_dict):
    pt = PyTensor(nested_dict)
    df = pd.DataFrame(nested_dict[dt.datetime(2022, 1, 1, 23, 59, 59)])
    assert (pt[1, "ETH", ["Val", "PNL"]].values == df["ETH"][["Val", "PNL"]].values).all()


def test_construct_by_nested_list(nested_dict):
    ls = pd.DataFrame(nested_dict[dt.datetime(2022, 1, 1, 23, 59, 59)]).values.tolist()
    pt = PyTensor(ls)
    assert pt[4].values.tolist() == ls[4]


def test_sort_index(slicing_dict):
    pt = PyTensor(slicing_dict)
    target1 = [[[30, 10], [40, 20]], [[3, 1], [4, 2]]]
    target2 = [[[1, 3], [2, 4]], [[10, 30], [20, 40]]]
    assert (pt.sort_index(axis=-1, reverse=True).values == target1).all()
    assert (pt.sort_index(axis=0, reverse=True).values == target2).all()


def test_slicing_id(slicing_dict):
    assert (PyTensor(slicing_dict)[0][0].values == PyTensor([10, 30]).values).all()


def test_slicing_fields(slicing_dict):
    pt = PyTensor(slicing_dict)
    assert pt["small"] == PyTensor({"first": [1, 3], "second": [2, 4]})
    target = [[[20, 40]], [[2, 4]]]
    assert (pt[["large", "small"], ["second"]].values == target).all()


def test_slice_without_header():
    pt = PyTensor([[0, 1, 2, 4], [0, 10, 20, 40], [0, 100, 200, 400]])
    assert (pt[1, 2:4].values == [20, 40]).all()


def test_bytes(nested_dict):
    pt = PyTensor(nested_dict)
    assert PyTensor.from_bytes(pt.to_bytes()) == pt
