import numpy as np
import pandas as pd
import pytest

from pyutil import PyTensor


@pytest.fixture
def nested_ls():
    return [[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]]


def test_swapaxes(nested_ls):
    xt = PyTensor(nested_ls)
    assert xt.swapaxes() == PyTensor([[[1, 2, 3], [10, 20, 30]], [[4, 5, 6], [40, 50, 60]]])
    assert xt.swapaxes(0, 2) == PyTensor([[[1, 10], [4, 40]], [[2, 20], [5, 50]],
                                         [[3, 30], [6, 60]]])


def test_fillna():
    xt = PyTensor([1, np.NaN, 3, np.NaN, 5])
    assert xt.fillna(fill_method="ffill") == PyTensor([1, 1, 3, 3, 5])
    assert xt.fillna(value=[5, 2, 3, 4, 1]) == PyTensor([1, 2, 3, 4, 5])
    assert xt.fillna(value=0) == PyTensor([1, 0, 3, 0, 5])
    xt = PyTensor([[[1, 2, np.NaN], [np.NaN, 3, 4]], [[5, np.NaN, 7], [np.NaN, 100, np.NaN]]])
    assert xt.fillna(fill_method="ffill") == PyTensor([[[1, 2, 2], [np.NaN, 3, 4]],
                                                      [[5, 5, 7], [np.NaN, 100, 100]]])
    assert xt.fillna(fill_method="bfill", axis=1) == PyTensor([[[1, 2, 4], [np.NaN, 3, 4]],
                                                              [[5, 100, 7], [np.NaN, 100, np.NaN]]])


def test_ema():
    xt = PyTensor([[0, 1, 2, 4], [0, 10, 20, 40], [0, 100, 200, 400]])
    target = pd.Series([0, 1, 2, 4]).ewm(alpha=2 / 3).mean().values.round(2)
    assert (xt.ema(com=0.5).values[0].round(2) == target).all()
    target = pd.Series([1, 10, 100]).ewm(alpha=0.5, adjust=False).mean()
    assert (xt.ema(span=3, axis=2, adjust=False)[:, 1].values == target).all()

    for adjust in [True, False]:
        alpha = 0.5
        value_random = np.random.rand(5000)
        xt = PyTensor(value_random).ema(alpha=alpha, adjust=adjust).values.round(3)
        target_value = pd.Series(value_random).ewm(alpha=alpha, ignore_na=True, adjust=adjust)
        assert (xt == target_value.mean().round(3)).all()


def test_shift(nested_ls):
    xt = PyTensor(nested_ls)
    assert xt.shift(1, -1) == PyTensor([[[np.NaN, 1, 2], [np.NaN, 4, 5]],
                                       [[np.NaN, 10, 20], [np.NaN, 40, 50]]])
    assert PyTensor([1, 2, 3, 4, 5]).shift(1) == PyTensor([np.NaN, 1, 2, 3, 4])
    assert xt.shift(1, -1, fill_method="roll") == PyTensor([[[3, 1, 2], [6, 4, 5]],
                                                           [[30, 10, 20], [60, 40, 50]]])
    target = PyTensor([[[10, 10, 10], [10, 10, 10]], [[1, 2, 3], [4, 5, 6]]])
    assert xt.shift(1, 0, fill_method="value", fill_value=10) == target
    shifted = xt.shift(1, -1, fill_method="matrix",
                       fill_value=np.array([[[0.15], [0.45]], [[0.1], [0.4]]]))
    assert shifted == PyTensor([[[0.15, 1, 2], [0.45, 4, 5]], [[0.1, 10, 20], [0.4, 40, 50]]])


def test_cumsum(nested_ls):
    xt = PyTensor(nested_ls)
    target = [[[1, 3, 6], [4, 9, 15]],
              [[10, 30, 60], [40, 90, 150]]]
    assert (xt.cumsum().values == target).all()
    target = [[[1, 2, 3], [5, 7, 9]],
              [[10, 20, 30], [50, 70, 90]]]
    assert (xt.cumsum(axis=1).values == target).all()


def test_cumprod(nested_ls):
    xt = PyTensor(nested_ls)
    target = [[[1, 2, 6], [4, 20, 120]],
              [[10, 200, 6000], [40, 2000, 120000]]]
    assert (xt.cumprod().values == target).all()
    target = [[[1, 2, 3], [4, 10, 18]],
              [[10, 20, 30], [400, 1000, 1800]]]
    assert (xt.cumprod(axis=1).values == target).all()


def test_diff(nested_ls):
    xt = PyTensor(nested_ls)
    target = PyTensor([[[np.NaN, 1, 1], [np.NaN, 1, 1]],
                      [[np.NaN, 10, 10], [np.NaN, 10, 10]]])
    assert xt.diff() == target
    target = PyTensor([[[-3, -3, -3], [np.NaN, np.NaN, np.NaN]],
                      [[-30, -30, -30], [np.NaN, np.NaN, np.NaN]]])
    assert xt.diff(-1, 1) == target
