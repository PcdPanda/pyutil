import pandas as pd

from pyutil._utils import try_import


def test_import_valid():
    pd_module = try_import("pandas")
    assert pd_module == pd


def test_import_nonexist():
    none_module = try_import("None")
    assert none_module is None
