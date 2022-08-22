import pytest

from pyutil import chunkify, flatten, listify, setify, uniquify


def test_chunkify():
    obj = list(range(1, 6))
    assert chunkify(obj, 1) == [[v] for v in obj]
    assert chunkify(obj, 2) == [[1, 2], [3, 4], [5]]
    assert chunkify("abcde", 2) == ["ab", "cd", "e"]
    assert chunkify(list(), 2) == list()


def test_chunkify_err():
    with pytest.raises(ValueError):
        chunkify(123, 1)


def test_flatten():
    obj = [range(3), ["ab", "cd"], [3, 4]]
    assert list(flatten(obj)) == [0, 1, 2, "ab", "cd", 3, 4]
    assert list(flatten("a")) == ["a"]


def test_listify():
    assert listify({1, 2, 3}) == [1, 2, 3]
    assert listify("abc") == ["abc"]
    assert listify(range(3)) == [0, 1, 2]


def test_setify():
    assert setify([1, 2, 1]) == {1, 2}
    assert setify("aba") == {"aba"}
    assert setify({"abc", "aba", "abc"}) == {"abc", "aba"}


def test_uniquify():
    assert uniquify([1, 2, 1]) == [1, 2]
    assert uniquify("aba") == ["aba"]
    assert sorted(uniquify(["abc", "aba", "abc"])) == ["aba", "abc"]
