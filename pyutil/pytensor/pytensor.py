from copy import deepcopy
import math
import pickle
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from numpy.testing import assert_equal
import pandas as pd

from pyutil import TensorLike


class PyTensor(object):
    """A multidimensional data structure similar to pandas.DataFrame.

    The class provides slicing, data manipulation, and calculation methods for multidimensional data
    and vectorization computation.

    Attributes
    ----------
    values: np.ndarray
        The numerical value stored by the PyTensor
    indexes: Iterable[np.ndarray]
        The index on each dimension
    shape: Tuple[int]:
        The shape of the PyTensor

    Parameters
    ----------
    values: TensorLike
        The numerical value to be stored in the PyTensor.
    indexes: Iterable[Iterable[Any]]
        The index on each dimension.

        The length of indexes should be the number of dimension.

        The length of each sub list of the indexes should be the length of each dimension.

    Examples
    --------
    >>> PyTensor([[[30, 10], [40, 20]], [[3, 1], [4, 2]]])
    PyTensor with shape:
    (2, 2, 2)
    indexes:
    [[0, 1], [0, 1], [0, 1]]
    values:
    [[[30 10]
    [40 20]]
    [[ 3  1]
    [ 4  2]]]

    >>> dt = {"s1": {"f": [1, 3], "g": [2, 4]}, "s2": {"f": [5, 9], "g": [7, 7]}}
    >>> pt = PyTensor(dt)  # Construct from nested data structure
    >>> pt
    PyTensor with shape:
    (2, 2, 2)
    indexes:
    ['s1' 's2'], ['f' 'g'], [0 1]
    values:
    [[[ 1  3]
    [ 2  4]]
    [[ 5  9]
    [7 7]]]


    >>> pt[["s2"], ["f"]]  # indexing by field
    PyTensor with shape:
    (1, 1, 2)
    indexes:
    [array(['s2'], dtype='<U2'), array(['f'], dtype='<U2'), array([0, 1])]
    values:
    [[[5 9]]]


    >>> df_dict = {"USDT": {"Val": 98, "Vol": 32}, "ETH": {"Val": 47, "Vol": 100}}
    >>> pt = PyTensor(pd.DataFrame(df_dict))
    >>> pt
    PyTensor with shape:
    (2, 2)
    indexes:
    ['Val' 'Vol'], ['USDT' 'ETH']
    values:
    [[ 98  47]
    [ 32 100]]

    >>> pt[1]  # indexing by id
    PyTensor with shape:
    (2,)
    indexes:
    ['USDT' 'ETH']
    values:
    [32 100]
    """

    def __init__(self, values: TensorLike, indexes: Iterable[Iterable[Any]] = list()):
        if isinstance(values, pd.DataFrame):  # construct from DataFrame
            self._values = np.array(values.copy(), copy=False)
            if not indexes:
                indexes = [np.array(values.index), np.array(values.columns)]
        elif isinstance(values, (int, float, np.number, np.ndarray)):
            self._values = np.array(values, copy=False)  # construct from a single element
        elif isinstance(values, PyTensor):  # move constructor
            self._values = values.values
            if not indexes:
                indexes = values.indexes.copy()
        else:  # Build XTenxor recursively from nested object
            self._values, indexes = self._load(values)
        # Make sure the index is np.ndarray
        self._indexes = [np.array(list(index)) for index in indexes]

    @classmethod
    def _load(cls, values: TensorLike) -> Tuple[np.ndarray, Iterable[np.ndarray]]:
        """Build a PyTensor recursively from nested data structure.
        The method will return the values and indexes for constructor
        """
        # Build the sub Xtensor recursively
        unaligned_values = dict()
        if isinstance(values, pd.Series):
            values = values.to_dict()
        if isinstance(values, dict):
            for key, val in values.items():
                unaligned_values[key] = cls(val)
        elif isinstance(values, Iterable) and not isinstance(values, str):
            for i, val in enumerate(values):
                unaligned_values[i] = cls(val)
        else:
            raise ValueError(f"Can only build PyTensor from TensorLike type, "
                             f"the given type is {type(values)}")

        # Make sure all sub PyTensor can be aligned and merged into one big PyTensor
        new_indexes = None
        for sub_xtensor in unaligned_values.values():
            if sub_xtensor.indexes:
                # Sort the sub pytensor so that different dimensions can be aligned correctly
                for i in range(len(sub_xtensor.shape)):
                    sub_xtensor.sort_index(axis=i)
                # New indexes are for merged pytensor, need to be checked with all sub XTensors'
                if new_indexes is None:
                    new_indexes = sub_xtensor.indexes
                elif sub_xtensor.indexes:
                    for i in range(len(sub_xtensor.shape)):  # Check every dimension
                        if not (new_indexes[i] == sub_xtensor.indexes[i]).all():
                            raise ValueError(f"There is a conflict within the nested PyTensor, "
                                             f"one has indexes {new_indexes} "
                                             f"while the other has {sub_xtensor.indexes}")

        # Merge sub xtensors together
        new_values = None
        new_first_dimension = list()
        for i, key in enumerate(sorted(unaligned_values.keys())):
            sub_xtensor = unaligned_values[key]
            new_first_dimension.append(key)
            if new_values is None:  # Initializae the new values
                new_values = np.ndarray(shape=[len(unaligned_values)] + list(sub_xtensor.shape),
                                        dtype=sub_xtensor.dtype)
            if new_values.dtype != sub_xtensor.dtype:  # Set the dtype
                new_values = new_values.astype(float)
                sub_xtensor = sub_xtensor.astype(float)
            new_values[i] = sub_xtensor.values
        if not new_indexes:  # If the user doesn't give any index, we generate default ones
            new_indexes = [[i for i in range(j)] for j in new_values.shape[1:]]
        new_indexes = [np.array(new_first_dimension)] + new_indexes
        return new_values, new_indexes

    @classmethod
    def _assert_constructible(cls, values: np.ndarray, indexes: Iterable[np.ndarray]) -> bool:
        """Check whether the input values and indexes are aligned with each other
        so that they can be used to construct a PyTensor"""
        if values.dtype == object:
            raise TypeError("The value's dtype can't be object")
        if indexes and values.shape != tuple([len(index) for index in indexes]):
            raise ValueError(f"The indexes doesn't fit the values shape, "
                             f"the values shape is {values.shape}, "
                             f"indexes shape={tuple([len(index) for index in indexes])}")

    def sort_index(self, axis: int = -1, reverse: bool = False, inplace: bool = False):
        """Sort the PyTensor based on the value of a given dimension

        Parameters
        ----------
        axis: int
            The dimension for sorting. By default, -1 is using last dimension.
        reverse: bool
            If False, sort ascendingly. Otherwise, sort descendingly.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None

        Examples
        --------
        >>> pt = PyTensor([[[30, 10], [40, 20]], [[3, 1], [4, 2]]])
        >>> pt.sort_index(axis=0)
        PyTensor with shape:
        (2, 2, 2)
        indexes:
        [array([0, 1]), array([0, 1]), array([0, 1])]
        values:
        [[[30 10]
        [40 20]]
        [[ 3  1]
        [ 4  2]]]
        """
        if inplace:
            if self._indexes:  # Only sort when there are valid indexes
                axis = axis % len(self.shape)
                # get the new values index based on the indexes
                index = np.array(self._indexes[axis]).argsort()
                self._indexes[axis].sort()
                if reverse:
                    index = index[::-1]
                    self._indexes[axis] = self._indexes[axis][::-1]
                sort_args = [slice(0, None, None) for i in range(len(self.shape))]
                sort_args[axis] = index
                self.values = self._values.__getitem__(tuple(sort_args))
        else:
            obj = self.copy()
            obj.sort_index(axis, reverse, True)
            return obj

    def __getitem__(self, indexes):
        if not self.shape:
            raise Exception("There is no dimension for slicing")
        # Initialize the index for getting item
        if isinstance(indexes, str) or not isinstance(indexes, Iterable):
            indexes = [indexes]
        j = 0
        new_values, new_indexes = self.values, list()
        for i, index in enumerate(indexes):  # Slicing multiple dimensions
            if isinstance(index, (Iterable, slice)) and not isinstance(index, str):
                # slicing, the index may be a list and a list of column
                if self.indexes:
                    if isinstance(index, slice):  # slicing case
                        index = [i for i in range(len(self.indexes[i]))].__getitem__(index)
                try:  # Assume the we're indexing by column names
                    field_index = {field: i for i, field in enumerate(self.indexes[i])}
                    index = [field_index[field] for field in index]
                except Exception:
                    pass
                # Use digit index to get new values and new indexes
                if self.indexes:
                    new_indexes.append(self.indexes[i].__getitem__(index))
                else:
                    new_indexes = list()
                new_values = new_values.__getitem__(tuple([slice(None, None, None)] * j + [index]))
                j += 1
            else:  # The index only be an int or a field name
                if self.indexes:
                    for k, field in enumerate(self.indexes[i]):
                        if field == index:
                            index = k
                new_values = new_values.__getitem__(tuple([slice(None, None, None)] * j + [index]))
        if self.indexes:
            new_indexes.extend(self.indexes[i + 1 :])
        else:
            new_indexes = list()
        ret = PyTensor(new_values, new_indexes)
        return ret

    def __setitem__(self, indexes: Union[Any, Iterable[Any]], new_values: TensorLike):
        """Add new field to the first dimension of PyTensor

        Args:
            fields (object): the name of the column
            new_values (object): the value
        """
        new_indexes = self.indexes
        if isinstance(new_values, PyTensor):
            new_values = new_values.values
        if indexes:
            sub_index = new_indexes[0].tolist()
            if indexes not in sub_index and (not isinstance(indexes, Iterable) or
                                             isinstance(indexes, str)):
                # Add new values
                indexes[0] = np.append(indexes[0], [indexes], axis=0)
                new_values = np.append(self.values, [new_values], axis=0)
            else:  # update existing values by index name
                index = sub_index.index(indexes) if indexes in sub_index else indexes
                _values = self.values
                _values[index] = new_values
                new_values = _values
        else:  # Add new values without editing indexes
            _values = self.values
            _values[indexes] = new_values
            new_values = _values
        self._assert_constructible(new_values, indexes)
        self._values = new_values
        self.indexes = indexes

    def __delitem__(self, index):
        """Delete field from certain dimension's index

        Args:
            index (object): The name of field
        """
        indexes = self.indexes
        values = self.values
        for i, field_index in enumerate(indexes[0]):
            if field_index == index:  # delete the sepcific field
                indexes[0] = np.delete(indexes[0], i, axis=0)
                values = np.delete(values, i, axis=0)
                break
        self._assert_constructible(values, indexes)
        self._indexes = indexes
        self.values = values

    def ema(self,
            com: Optional[float] = None,
            span: Optional[float] = None,
            halflife: Optional[float] = None,
            alpha: Optional[float] = None,
            adjust: bool = True,
            axis: int = -1,
            inplace: bool = False):
        """Use vectorization method to calculate exponential moving average based on the formula
        given by pandas.DataFrame.ewm on a given dimension

        Exactly one parameter: ``com``, ``span``, ``halflife``, or ``alpha`` must be provided.

        Parameters
        ----------
        com: Optional[float]
            Specify decay in terms of center of mass, alpha = 1 / (1 + com)

        span: Optional[float]
            Specify decay in terms of span, alpha = 2 / (span + 1)

        halflife: Optional[float]
            Specify decay in terms of half-life, alpha = 1 - exp(-ln(2) / halflife)

        alpha: Optional[float]
            Specify smoothing factor directly

        adjust: bool
            Divide by decaying adjustment factor in beginning periods to account
            for imbalance in relative weightings (viewing EWMA as a moving average).

            - When ``adjust=True``,
                y_{t} = (x_t + (1 - alpha) * x_{t-1} + ... + (1 - alpha) ^ t * x_0)
                / (1 + (1-alpha) + ... + (1 - alpha) ^ t)
            - When ``adjust=False``, y_{t} = (1 - alpha) * y_{t-1} + alpha * x_{t}
        axis: int
            The dimension for ema calculation.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.
        """
        axis = axis % len(self.shape)
        if inplace:
            if com == span == halflife == alpha is None:
                raise ValueError("The alpha is not specified")
            elif com is not None and span == halflife == alpha is None:
                alpha = 1 / (1 + com)
            elif span is not None and com == halflife == alpha is None:
                alpha = 2 / (span + 1)
            elif halflife is not None and com == span == alpha is None:
                alpha = 1 - math.exp(-math.log(2) / halflife)
            elif not (alpha is not None and com == span == halflife is None):
                raise ValueError("Parameters com, span, halflife and alpha are mutually exclusive")

            # Swap the axis of calculation to the last dimension
            new_values = self.values.swapaxes(-1, axis).astype(float)
            shape = new_values.shape
            new_values = new_values.reshape(int(np.prod(shape[:-1])), shape[-1])
            if np.isnan(new_values).any():
                raise ValueError("EMA calculation for missing value is not supported yet")

            # Calculate scale = [(1 - alpha) ^ t, ... (1 - alpha)]
            block_size = int(np.log(np.finfo(float).tiny) / np.log(1 - alpha))
            scale_coeff = np.power(1 - alpha, np.arange(block_size + 1))[::-1]

            # Calculate ema block by block to avoid numeric problem caused by many multiplications
            def ema_block(block: np.ndarray):
                # Sometimes the block may not be aligned well
                valid_scale_coeff = scale_coeff[:block.shape[-1]]
                alpha_coeff = np.ones(shape=valid_scale_coeff.shape)
                if not adjust:
                    alpha_coeff[1:] = alpha
                # Calculate block = [(1 - alpha) ^ t * x_0, ... (1 - alpha) * x_t * alpha]
                block = alpha_coeff * block * valid_scale_coeff
                # Sum everything up then we get not adjusted ema for the block
                block = np.cumsum(block, axis=-1) / valid_scale_coeff
                return block

            for i in range(0, new_values.shape[-1], block_size):
                if i == 0:
                    new_values[:, :block_size] = ema_block(new_values[:, :block_size])
                else:
                    block = new_values[:, i - 1 : i + block_size]
                    new_values[:, i : i + block_size] = ema_block(block)[:, 1:]
            if adjust:
                new_values /= np.cumsum(np.power(1 - alpha, np.arange(new_values.shape[-1])), -1)
            self.values = new_values.reshape(shape).swapaxes(-1, axis)
        else:
            obj = self.copy()
            obj.ema(com, span, halflife, alpha, adjust, axis, True)
            return obj

    def fillna(self, fill_method: str = "value", value: Optional[Union[float, int]] = None,
               axis: int = -1, inplace: bool = False):
        """Shift the values of PyTensor along a given dimension and fill the NaN value by a given way.

        Parameters
        ----------
        fill_method: str
            The method to fill the NaN value after shifting,
            should be one of ``"value" / "matrix" / "roll"``.

            - When ``fill_method="value"``, will use the parameter value to fill all NaN.
            - When ``fill_method="ffill"`` or ``fill_method="bfill"``, will use the previous / next
                valid value along the dimension given by parameter axis to fill.
        value: Optional[Union[float, int]]
            The value to fill NaN, only valid when ``fill_method="value"``.
        axis: int
            The dimension for shifting.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.

        Examples
        --------
        >>> fill_value = range(10, 16)
        >>> PyTensor([1, np.NaN, 3, np.NaN, np.NaN, 5]).fillna(value=fill_value)
        PyTensor with shape:
        (6,)
        indexes:
        [0 1 2 3 4 5]
        values:
        [ 1. 11.  3. 13. 14.  5.]
        >>> pt = PyTensor([[np.NaN, np.NaN, np.NaN], [2, np.NaN, 8], [np.NaN, 0, np.NaN]])
        >>> pt.fillna(fill_method = "bfill", axis=0)
        PyTensor with shape:
        (3, 3)
        indexes:
        [0 1 2], [0 1 2]
        values:
        [[ 2.  0.  8.]
        [ 2.  0.  8.]
        [nan  0. nan]]
        """
        if inplace:
            if fill_method == "value":
                if value is None:
                    raise ValueError("NaN value cannot be filled by None")
                if not isinstance(value, Iterable):
                    self.values = np.nan_to_num(self.values, nan=value)
                else:
                    value = value.values if isinstance(value, PyTensor) else np.array(value)
                    if self.values.shape != value.shape:
                        raise ValueError(f"The shape of given value is not compitable "
                                         f"the given value shape is {value.shape} "
                                         f"the PyTensor shape is {self.values.shape}")
                    self.values = np.where(~np.isnan(self.values), self.values, value)
            else:  # Use rolling method to fill
                axis = axis % len(self.shape)
                new_values = self.values.swapaxes(axis, -1)
                new_shape = new_values.shape
                if len(self.shape) > 1:
                    new_values = new_values.reshape(int(np.prod(new_shape[:-1])), new_shape[-1])
                else:
                    new_values = new_values.reshape(1, new_shape[-1])
                if fill_method == "bfill":
                    new_values = new_values[:, ::-1]
                elif fill_method != "ffill":
                    raise ValueError(f"Fill_method must be in {'value', 'ffill', 'bfill'}, "
                                     f"the given one is {fill_method}")
                mask = np.isnan(new_values)
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                new_values = new_values[np.arange(idx.shape[0])[:, None], idx]
                if fill_method == "bfill":
                    new_values = new_values[:, ::-1]
                self.values = new_values.reshape(new_shape).swapaxes(axis, -1)
        else:
            obj = self.copy()
            obj.fillna(fill_method, value, axis, True)
            return obj

    def shift(self,
              periods: int = 0,
              axis: int = -1,
              fill_method: str = "value",
              fill_value: Union[float, np.ndarray] = np.NaN,
              inplace: bool = False):
        """Shift the values of PyTensor along a given dimension and fill the NaN value by a given method.

        Parameters
        ----------
        periods: int
            Number of periods to shift. Can be positive or negative.
        axis: int
            The dimension for shifting.
        fill_method: str
            The way to fill the NaN value after shifting,
            should be one of ``"value" / "matrix" / "roll"``.
        fill_value: Union[float, np.ndarray]
            The value to fill NaN when ``fill_method="value"`` or ``fill_method="matrix"``.

            - When ``fill_method="value"``, fill_value should be a float to fill every NaN value.
            - When ``fill_method="matrix"``, fill_value should be a matrix to fill NaN block
                caused by shifting.

        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.

        Examples
        --------
        >>> PyTensor([1, 2, 3, 4, 5]).shift(2)
        PyTensor with shape:
        (5,)
        indexes:
        [0 1 2 3 4]
        values:
        [nan nan  1.  2.  3.]
        >>> pt = PyTensor([[1, 2, 3], [4, 5, 6]])
        >>> pt.shift(1, fill_method="value", fill_value=0, axis=1)
        PyTensor with shape:
        (2, 3)
        indexes:
        [0 1], [0 1 2]
        values:
        [[10. 1. 2.]
        [10. 4. 5.]]
        >>> fill_value = np.array([[7, 8, 9]])
        >>> pt.shift(-1, fill_method="matrix", fill_value=fill_value, axis=0)
        PyTensor with shape:
        (2, 3)
        indexes:
        [0 1], [0 1 2]
        values:
        [[4. 5. 6.]
        [7. 8. 9.]]
        """
        if inplace:
            axis = axis % len(self.shape)
            shape = list(self.shape)
            cut = [slice(None, None, None)] * len(shape)
            if abs(periods) >= shape[axis]:
                raise ValueError(f"The absoluate value of periods must be smaller than "
                                 f"shape[{axis}]={shape[axis]}, the given input is {periods}")
            if not periods:
                periods = None
            cut[axis] = slice(None, -periods, None) if periods > 0 else slice(-periods, None, None)
            values_shifted = self.values.__getitem__(tuple(cut))
            shape[axis] = abs(periods)
            if fill_method == "value":
                if not isinstance(fill_value, (int, float)):
                    raise ValueError(f"When fill_method is value, fill_value must be int or float, "
                                     f"given fill_value type is {type(fill_value)}")
                values_filled = np.zeros(shape=shape) + fill_value
            elif fill_method == "roll":
                if periods > 0:
                    cut[axis] = slice(-periods, None, None)
                else:
                    cut[axis] = slice(None, -periods, None)
                values_filled = self.values.__getitem__(tuple(cut))
            elif fill_method == "matrix":
                values_filled = fill_value
            else:
                raise ValueError(f"fill_method must be in roll, value, matrix, "
                                 f"the given input is {fill_method}")
            if list(values_filled.shape) != shape:
                raise ValueError(f"The given padding value's shape is not compitable, "
                                 f"padding value shape is {values_filled.shape}, "
                                 f"the PyTensor shape is {shape}")
            values_filled = values_filled.swapaxes(0, axis)
            values_shifted = values_shifted.swapaxes(0, axis)
            shape = list(values_shifted.shape)
            shape[0] += values_filled.shape[0]
            new_values = np.empty(shape=shape)
            if periods > 0:
                new_values[: values_filled.shape[0]] = values_filled
                new_values[values_filled.shape[0] :] = values_shifted
            else:
                new_values[: values_shifted.shape[0]] = values_shifted
                new_values[values_shifted.shape[0] :] = values_filled
            self.values = new_values.swapaxes(0, axis)
        else:
            obj = self.copy()
            obj.shift(periods, axis, fill_method, fill_value, True)
            return obj

    def swapaxes(self, axis1: int = 0, axis2: int = 1, inplace: bool = False):
        """Exchange two dimenstions of the PyTensor.

        Parameters
        ----------
        axis1: int
            the first dimension to be changed, -1 means last dimension.
        axis2: int
            the first dimension to be changed, -1 means last dimension.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.

        Examples
        --------
        >>> pt = PyTensor([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])
        >>> pt.swapaxes(0, 2)
        PyTensor with shape:
        (3, 2, 2)
        indexes:
        [0 1 2], [0 1], [0 1]
        values:
        [[[ 1  7]
        [ 4 10]]
        [[ 2  8]
        [ 5 11]]
        [[ 3  9]
        [ 6 12]]]
        """
        if inplace:
            axis1 = axis1 % len(self.shape)
            axis2 = axis2 % len(self.shape)
            new_values = self._values.swapaxes(axis1, axis2)
            new_indexes = self.indexes.copy()
            if self.indexes:
                new_indexes[axis1], new_indexes[axis2] = self.indexes[axis2], self.indexes[axis1]
            self._assert_constructible(new_values, new_indexes)
            self._values, self._indexes = new_values, new_indexes
        else:
            obj = self.copy()
            obj.swapaxes(axis1, axis2, True)
            return obj

    def cumsum(self, axis: int = -1, inplace: bool = False):
        """Calculate cumulative sum on a given axis.

        Parameters
        ----------
        axis: int
            The dimension for cumulative sum calculation.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.

        Examples
        --------
        >>> PyTensor([[30, 10, 1], [40, 20, 2]]).cumprod(axis=1)
        PyTensor with shape:
        (2, 3)
        indexes:
        [0 1], [0 1 2]
        values:
        [[30 40 41]
        [40 60 62]]
        """
        axis = axis % len(self.shape)
        if inplace:
            self.values = np.cumsum(self.values, axis)
        else:
            obj = self.copy()
            obj.cumsum(axis, True)
            return obj

    def cumprod(self, axis: int = -1, inplace: bool = False):
        """Calculate cumulative product on a given axis.

        Parameters
        ----------
        axis: int
            The dimension for cumulative product calculation.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.

        Examples
        --------
        >>> PyTensor([[30, 10, 1], [40, 20, 2]]).cumprod(axis=1)
        PyTensor with shape:
        (2, 3)
        indexes:
        [0 1], [0 1 2]
        values:
        [[  30  300  300]
        [  40  800 1600]]
        """
        if inplace:
            axis = axis % len(self.shape)
            self.values = np.cumprod(self.values, axis)
        else:
            obj = self.copy()
            obj.cumprod(axis, True)
            return obj

    def diff(self, periods: int = 1, axis: int = -1, inplace: bool = False):
        """Calculate difference of PyTensor on a given axis.

        Parameters
        ----------
        periods: int
            Periods to shift for calculating difference, accepts negative values.
        axis: int
            The dimension for difference calculation.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.

        Examples
        --------
        >>> pt = PyTensor([[30, 10], [40, 20]])
        >>> pt.diff(axis=1)
        PyTensor with shape:
        (2, 2)
        indexes:
        [0 1], [0 1]
        values:
        [[20. nan]
        [20. nan]]

        >>> PyTensor([1, 3, 11, 7, 15]).diff(-2)
        PyTensor with shape:
        (5,)
        indexes:
        [0 1 2 3 4]
        values:
        [-10.  -4.  -4.  nan  nan]
        """
        if inplace:
            axis = axis % len(self.shape)
            shifted_values = self.shift(periods, axis=axis).values
            self.values = self.values.astype(shifted_values.dtype) - shifted_values
        else:
            obj = self.copy()
            obj.diff(periods, axis, True)
            return obj

    def round(self, decimals: int = 0, to: str = "", inplace: bool = False):
        """Round the PyTensor to given precision.

        Parameters
        ----------
        decimals: int
            Number of decimal places to round to.
        to: str
            If the value is floor, then round to the floor.
            If the value is ceil, then round to the ceiling.
            Otherwise, will do a normal rounding.
        inplace: bool
            If False, return a copy. Otherwise, do operation inplace and return None.

        Examples
        --------
        >>> PyTensor([1.23, 1.282, 4, 5.7779]).round(2)
        PyTensor with shape:
        (4,)
        indexes:
        [array([0, 1, 2, 3])]
        values:
        [1.23 1.28 4.   5.78]

        >>> PyTensor([[1.23, 1.28], [4, 5.7779]]).round(1, "ceil")
        PyTensor with shape:
        (2, 2)
        indexes:
        [array([0, 1]), array([0, 1])]
        values:
        [[1.3 1.3]
        [4.  5.8]]
        """
        if to and to not in {"floor", "ceil"}:
            raise ValueError(f"Parameter to must be in {{\"floor\", \"ceil\", None}}, "
                             f"given {to}")
        if not isinstance(decimals, int) or decimals < 0:
            raise ValueError(f"Parameter decimals must be a non-negative integer, given {decimals}")
        if inplace:
            if to == "ceil":
                self.values = self.values * (10 ** decimals)
                self.values = np.round(np.ceil(self.values) / (10 ** decimals), decimals)
            elif to == "floor":
                self.values = self.values * (10 ** decimals)
                self.values = np.round(np.floor(self.values) / (10 ** decimals), decimals)
            else:
                self.values = np.round(self.values, decimals)
        else:
            obj = self.copy()
            obj.round(decimals, to, True)
            return obj

    def copy(self):
        """Create a deep copyed Xtensor.

        Examples
        --------
        >>> PyTensor({"v1": [1, 3], "v2": [2, 4]}).copy()
        PyTensor with shape:
        (2, 2)
        indexes:
        [array(['v1', 'v2'], dtype='<U2'), array([0, 1])]
        values:
        [[1 3]
        [2 4]]
        """
        res = PyTensor(self._values.copy(), deepcopy(self._indexes))
        return res

    def to_dict(self) -> Dict[Any, "PyTensor"]:
        """Transform PyTensor into a dict, where the keys are the index names of the first dimension,
        and the values are sub Xtensors.

        Examples
        --------
        >>> PyTensor({"v1": [1, 2, 3], "v2": [4, 5, 6]}).to_dict()
        {'v1': PyTensor with shape:
               (3,)
               indexes:
               [array([0, 1, 2])]
               values:
               [1 2 3],
        'v2': PyTensor with shape:
              (3,)
              indexes:
              [array([0, 1, 2])]
              values:
              [4 5 6]}
        """
        ret = dict()
        for key in self.indexes[0]:
            ret[key] = self[key]
        return ret

    def to_df(self) -> pd.DataFrame:
        """Transform PyTensor with two dimension into a pandas.DataFrame.

        Examples
        --------
        >>> PyTensor({"v1": [1, 3], "v2": [2, 4]}).to_df()
            0  1
        v1  1  3
        v2  2  4
        """
        if len(self.shape) != 2:
            raise ValueError("The Xtensor to be transformed must have 2 dimensions")
        return pd.DataFrame(self._values, index=self.indexes[0], columns=self.indexes[-1])

    def to_bytes(self) -> bytes:
        """Serialize the PyTensor to binary.

        Examples
        --------
        >>> xt_bytes = PyTensor([1, 2, 3]).to_bytes()
        """
        data = self.values.tobytes()
        meta_info = {
            "indexes": self.indexes,
            "dtype": self.dtype,
            "shape": self.shape,
            "size": len(data),
        }
        meta_info_pkl = pickle.dumps(meta_info)
        meta_info_size = len(meta_info_pkl).to_bytes(32, byteorder="big")
        return meta_info_size + meta_info_pkl + data

    @classmethod
    def from_bytes(cls, binary: bytes) -> "PyTensor":
        """Deserialize PyTensor from bytes.

        Parameters
        ----------
        binary: bytes
            Serialized PyTensor in binary format.

        Examples
        --------
        >>> xt_bytes = PyTensor([1, 2, 3]).to_bytes()
        >>> PyTensor.from_bytes(xt_bytes)
        PyTensor with shape:
        (3,)
        indexes:
        [array([0, 1, 2])]
        values:
        [1 2 3]
        """
        meta_info_size = int.from_bytes(binary[:32], byteorder="big")
        meta_info = pickle.loads(binary[32 : 32 + meta_info_size])
        indexes = meta_info["indexes"]
        dtype = meta_info["dtype"]
        shape = meta_info["shape"]
        size = meta_info["size"]
        values = np.ndarray(
            dtype=dtype,
            shape=shape,
            buffer=binary[32 + meta_info_size : 32 + meta_info_size + size],
        )
        return cls(values=values, indexes=indexes)

    @property
    def dtype(self):
        return self._values.dtype

    @dtype.setter
    def dtype(self, dtype):
        self._values = self.values.astype(dtype)

    @property
    def shape(self):
        return self._values.shape

    @property
    def nbytes(self):
        return self._values.nbytes

    @property
    def indexes(self):
        return list(self._indexes)

    @indexes.setter
    def indexes(self, indexes: Iterable[Iterable[Any]]):
        indexes = list(indexes)
        self._assert_constructible(self._values, indexes)
        self._indexes = tuple([np.array(index) for index in enumerate(indexes)])

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values: object):
        self._assert_constructible(values, self.indexes)
        self._values = np.array(values)

    @property
    def T(self):
        """Exchange the first dimension with the last dimension"""
        return self.swapaxes(0, -1)

    def astype(self, dtype: type):
        """Represent the PyTensor as another dtype"""
        res = PyTensor(self._values.astype(dtype), self._indexes)
        return res

    def __repr__(self):
        indexes = ", ".join([f"{index}" for index in self.indexes])
        return (f"PyTensor with shape:\n{self.shape}\n"
                f"indexes:\n{indexes}\nvalues:\n{self.values}")

    def __eq__(self, other):
        try:
            assert self.shape == other.shape
            assert_equal(self.values, other.values)
            if self.indexes and other.indexes:
                for i in range(len(self.shape)):
                    assert_equal(self.indexes[i], other.indexes[i])
        except Exception:
            return False
        return True

    def _assert_compatible(self, other):
        """Check the format of self and other PyTensor so that
        they're compatible for arithmetic operation"""
        if isinstance(other, np.ndarray):
            other = PyTensor(other)
            if other.shape == self.shape:
                return other
        elif isinstance(other, int) or isinstance(other, float):
            return PyTensor(np.ones(shape=self.shape) * other)
        if not isinstance(other, PyTensor):
            raise ValueError("Can't construct a PyTensor based on the input")
        try:
            assert self.shape == other.shape
            if self.indexes and other.indexes:
                for i in range(len(self.shape)):
                    assert_equal(self.indexes[i], other.indexes[i])
        except Exception:
            raise ValueError(f"The two XTensors are not compatible for arithmetic operation\n"
                             f"self.indexes={self.indexes}, other.indexes={other.indexes}\n"
                             f"self.shape={self.shape}, other.shape={other.shape}")
        return other

    def __add__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values + other.values
        return PyTensor(values, self._indexes)

    def __radd__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = other.values + self.values
        return PyTensor(values, self._indexes)

    def __sub__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values - other.values
        return PyTensor(values, self._indexes)

    def __rsub__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = other.values - self.values
        return PyTensor(values, self._indexes)

    def __mul__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values * other.values
        return PyTensor(values, self._indexes)

    def __rmul__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = other.values * self.values
        return PyTensor(values, self._indexes)

    def __truediv__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values / other.values
        return PyTensor(values, self._indexes)

    def __rtruediv__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = other.values / self.values
        return PyTensor(values, self._indexes)

    def __gt__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values > other.values
        return PyTensor(values, self._indexes)

    def __lt__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values < other.values
        return PyTensor(values, self._indexes)

    def __ge__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values >= other.values
        return PyTensor(values, self._indexes)

    def __le__(self, other):
        other = PyTensor(self._assert_compatible(other))
        values = self.values <= other.values
        return PyTensor(values, self._indexes)

    def __invert__(self):
        return PyTensor(~self.values, self._indexes)

    def __abs__(self):
        return PyTensor(abs(self.values), self._indexes)
