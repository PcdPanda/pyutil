import numpy as np
import pandas as pd
import pytest

from pyutil import XTensor


@pytest.fixture
def nested_ls():
    return [[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]]


def test_swapaxes(nested_ls):
    xt = XTensor(nested_ls)
    assert xt.swapaxes() == XTensor([[[1, 2, 3], [10, 20, 30]], [[4, 5, 6], [40, 50, 60]]])
    assert xt.swapaxes(0, 2) == XTensor([[[1, 10], [4, 40]], [[2, 20], [5, 50]],
                                         [[3, 30], [6, 60]]])


def test_fillna():
    xt = XTensor([1, np.NaN, 3, np.NaN, 5])
    assert xt.fillna(method="ffill") == XTensor([1, 1, 3, 3, 5])
    assert xt.fillna(value=[5, 2, 3, 4, 1]) == XTensor([1, 2, 3, 4, 5])
    assert xt.fillna(value=0) == XTensor([1, 0, 3, 0, 5])
    xt = XTensor([[[1, 2, np.NaN], [np.NaN, 3, 4]], [[5, np.NaN, 7], [np.NaN, 100, np.NaN]]])
    assert xt.fillna(method="ffill") == XTensor([[[1, 2, 2], [np.NaN, 3, 4]],
                                                [[5, 5, 7], [np.NaN, 100, 100]]])
    assert xt.fillna(method="bfill", axis=1) == XTensor([[[1, 2, 4], [np.NaN, 3, 4]],
                                                         [[5, 100, 7], [np.NaN, 100, np.NaN]]])


def test_ema():
    xt = XTensor([[0, 1, 2, 4], [0, 10, 20, 40], [0, 100, 200, 400]])
    target = pd.Series([0, 1, 2, 4]).ewm(alpha=2 / 3).mean().values.round(2)
    assert (xt.ema(com=0.5).values[0].round(2) == target).all()
    target = pd.Series([1, 10, 100]).ewm(alpha=0.5, adjust=False).mean()
    assert (xt.ema(span=3, axis=2, adjust=False)[:, 1].values == target).all()

    for adjust in [True, False]:
        alpha = 0.5
        value_random = np.random.rand(5000)
        xt = XTensor(value_random).ema(alpha=alpha, adjust=adjust).values.round(3)
        target_value = pd.Series(value_random).ewm(alpha=alpha, ignore_na=True, adjust=adjust)
        assert (xt == target_value.mean().round(3)).all()


def test_shift(nested_ls):
    xt = XTensor(nested_ls)
    assert xt.shift(1, -1) == XTensor([[[np.NaN, 1, 2], [np.NaN, 4, 5]],
                                       [[np.NaN, 10, 20], [np.NaN, 40, 50]]])
    assert XTensor([1, 2, 3, 4, 5]).shift(1) == XTensor([np.NaN, 1, 2, 3, 4])
