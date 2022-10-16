from copy import deepcopy
import math
import pickle
from typing import Any, Dict, Iterable, Tuple, Union


import numpy as np
from numpy.testing import assert_equal
import pandas as pd

from pyutil import TensorLike


class XTensor(object):
    """A multidimensional data structure similar to pandas.DataFrame.
    The class aims at providing data manipulation methods for multidimensional data
    and vectorization computation.
    """

    def __init__(self, values: TensorLike, indexes: Iterable[Iterable[Any]] = list()):
        """Build a data tensor from tensor like value. The value should be aligned
        so that it can be stored in a numpy.ndarray

        Parameters
        ==========
        values: TensorLike
            The key name of the shared memory for the lock free queue
        indexes: Iterable[Iterable[Any]]
            The index on each dimension.
            The length of indexes should be the number of dimension
            The length of each sub list of the indexes should be the length of each dimension

        Examples
        ========
        >>> 123
        """
        if isinstance(values, pd.DataFrame):  # construct from DataFrame
            self._values = np.array(values.copy(), copy=False)
            if not indexes:
                indexes = [np.array(values.index), np.array(values.columns)]
        elif isinstance(values, (int, float, np.number, np.ndarray)):
            self._values = np.array(values, copy=False)  # construct from a single element
        elif isinstance(values, XTensor):  # move constructor
            self._values = values.values
            if not indexes:
                indexes = values.indexes.copy()
        else:  # Build XTenxor recursively from nested object
            self._values, indexes = self._load(values)
        # Make sure the index is np.ndarray
        self._indexes = [np.array(list(index)) for index in indexes]

    @classmethod
    def _load(cls, values: TensorLike) -> Tuple[np.ndarray, Iterable[np.ndarray]]:
        """Build a XTensor recursively from nested data structure.
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
            raise ValueError(f"Can only build XTensor from TensorLike type, "
                             f"the given type is {type(values)}")

        # Make sure all sub XTensor can be aligned and merged into one big XTensor
        new_indexes = None
        for sub_xtensor in unaligned_values.values():
            if sub_xtensor.indexes:
                # Sort the sub xtensor so that different dimensions can be aligned correctly
                for i in range(len(sub_xtensor.shape)):
                    sub_xtensor.sort_index(axis=i)
                # New indexes are for merged xtensor, need to be checked with all sub XTensors'
                if new_indexes is None:
                    new_indexes = sub_xtensor.indexes
                elif sub_xtensor.indexes:
                    for i in range(len(sub_xtensor.shape)):  # Check every dimension
                        if not (new_indexes[i] == sub_xtensor.indexes[i]).all():
                            raise ValueError(f"There is a conflict within the nested XTensor, "
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
    def assert_constructible(cls, values: np.ndarray, indexes: Iterable[np.ndarray]) -> bool:
        """Check whether the input values and indexes are aligned with each other
        so that they can be used to construct a XTensor

        Parameters
        ==========
        values: np.ndarray
            The key name of the shared memory for the lock free queue
        indexes: Iterable[np.ndarray]
            The size of the queue, should be power of 2

        Examples
        ========
        >>> queue = SharedLockFreeQueue("my_queue", size=2 ** 28)
        >>> queue.delete("my_queue")
        """
        if values.dtype == object:
            raise TypeError("The value's dtype can't be object")
        if indexes and values.shape != tuple([len(index) for index in indexes]):
            raise ValueError(f"The indexes doesn't fit the values shape, "
                             f"the values shape is {values.shape}, "
                             f"indexes shape={tuple([len(index) for index in indexes])}")

    def sort_index(self, axis: int = -1, reverse: bool = False, inplace: bool = False):
        """Sort the data based on index value

        Args:
            axis (int): the axis used for sorting, default is the last
            reverse (bool): whether sort reversely
            inplace (bool):
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
        ret = XTensor(new_values, new_indexes)
        return ret

    def __setitem__(self, indexes: Union[Any, Iterable[Any]], new_values: TensorLike):
        """Add new field to the first dimension of XTensor

        Args:
            fields (object): the name of the column
            new_values (object): the value
        """
        new_indexes = self.indexes
        if isinstance(new_values, XTensor):
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
        self.assert_constructible(new_values, indexes)
        self._values = new_values
        self.indexes = indexes

    def __delitem__(self, index):
        """Delete field from certain dimension's index

        Args:
            index (object): The name of field
        """
        indexes = self.indexes
        values = self.values
        for i, head in enumerate(indexes[0]):
            if head == index:  # delete the sepcific field
                indexes[0] = np.delete(indexes[0], i, axis=0)
                values = np.delete(values, i, axis=0)
                break
        self.assert_constructible(values, indexes)
        self._indexes = indexes
        self.values = values

    def ema(self, com: float = 0, span: float = 0, halflife: float = 0, alpha: float = 0,
            adjust: bool = True, ignore_na: bool = True, axis: int = -1, inplace: bool = False):
        """Calculate ema

        Args:
            com (float): Optionally specify decay in terms of center of mass
            span (float): Optionally specify decay in terms of span
            halflife (float): Optionally specify decay in terms of half-life
            alpha (float): Optionally specifiy the parameters for ema calculation
            adjust (bool): Divide by decaying adjustment factor in beginning periods
            ignore_na (bool): Whether to ignore the nan
            axis (int=-1): The axis for the operation
            inplace (bool):
        """
        axis = axis % len(self.shape)
        if inplace:
            if com == span == halflife == alpha == 0:
                raise ValueError("The alpha is not specified")
            elif com != 0 and span == halflife == alpha == 0:
                alpha = 1 / (1 + com)
            elif span != 0 and com == halflife == alpha == 0:
                alpha = 2 / (span + 1)
            elif halflife != 0 and com == span == alpha == 0:
                alpha = 1 - math.exp(-math.log(2) / halflife)
            elif not (alpha != 0 and com == span == halflife == 0):
                raise ValueError("Parameters com, span, halflife and alpha are mutually exclusive")

            # Swap the axis of calculation to the last dimension
            new_values = self.values.swapaxes(-1, axis).astype(float)
            shape = new_values.shape
            new_values = new_values.reshape(int(np.prod(shape[:-1])), shape[-1])
            if np.isnan(new_values).any():
                raise ValueError("EMA calculation for missing value is not supported yet")

            # Calculate ema block by block to avoid numeric problem caused by many multiplications
            def ema_block(block: np.ndarray):
                scale_fac = np.power(1 - alpha, np.arange(block.shape[-1]))[::-1]
                alpha_fac = np.ones(shape=scale_fac.shape)
                if not adjust:
                    alpha_fac[1:] = alpha  # [1, alpha, alpha, ...]
                block = alpha_fac * block * scale_fac
                block = np.cumsum(block, axis=-1) / scale_fac
                return block

            block_size = int(np.log(np.finfo(float).tiny) / np.log(1 - alpha))
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
            obj.ema(com, span, halflife, alpha, adjust, ignore_na, axis, True)
            return obj

    def fillna(self, value: object = None, method: str = "value",
               axis: int = -1, inplace: bool = False):
        """Fill the nan in the values

        Args:
            val (object): The value to be used for filling
            method (str): Filling method, can be value / ffill / bfill
            axis (int): The axis for filling when using prev or after value
            inplace (bool):
        """
        if inplace:
            if method == "value":
                if value is None:
                    raise ValueError("NaN value cannot be filled by None")
                if not isinstance(value, Iterable):
                    self.values = np.nan_to_num(self.values, nan=value)
                else:
                    value = value.values if isinstance(value, XTensor) else np.array(value)
                    if self.values.shape != value.shape:
                        raise ValueError(f"The shape of given value is not compitable "
                                         f"the given value shape is {value.shape} "
                                         f"the XTensor shape is {self.values.shape}")
                    self.values = np.where(~np.isnan(self.values), self.values, value)
            else:  # Use rolling method to fill
                axis = axis % len(self.shape)
                new_values = self.values.swapaxes(axis, -1)
                new_shape = new_values.shape
                if len(self.shape) > 1:
                    new_values = new_values.reshape(int(np.prod(new_shape[:-1])), new_shape[-1])
                else:
                    new_values = new_values.reshape(1, new_shape[-1])
                if method == "bfill":
                    new_values = new_values[:, ::-1]
                elif method != "ffill":
                    raise ValueError(f"Parameter method must be in {'value', 'ffill', 'bfill'}, "
                                     f"the given one is {method}")
                mask = np.isnan(new_values)
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                new_values = new_values[np.arange(idx.shape[0])[:, None], idx]
                if method == "bfill":
                    new_values = new_values[:, ::-1]
                self.values = new_values.reshape(new_shape).swapaxes(axis, -1)
        else:
            obj = self.copy()
            obj.fillna(value, method, axis, True)
            return obj

    def shift(self, step: int = 0, axis: int = -1, fill_type: str = "value",
              inplace: bool = False, fillna: float = np.NaN):
        """Shift operation similar to pandas

        Args:
            step (int): the step for shifting, can't exceed the length of this axis
            axis (int): the axis on which shifting is perfomred
            fill_type (str): filling type

                'value': fill the nan with specific value

                'roll': fill the nan with rolling value

                'matrix': fill the nan with an input matrix

            inplace (bool):
            fillna (float): the value/matrix used for filling, depends on the fill_type
        """
        if inplace:
            axis = axis % len(self.shape)
            shape = list(self.shape)
            cut = [slice(None, None, None)] * len(shape)
            if abs(step) >= shape[axis]:
                raise ValueError(f"The absoluate value of step must be smaller than "
                                 f"shape[{axis}]={shape[axis]}, the given input is {step}")
            if not step:
                step = None
            cut[axis] = slice(None, -step, None) if step > 0 else slice(-step, None, None)
            value_shifting = self.values.__getitem__(tuple(cut))
            shape[axis] = abs(step)
            if fill_type == "value":
                value_padding = np.zeros(shape=shape) + fillna
            elif fill_type == "roll":
                cut[axis] = slice(-step, None, None) if step > 0 else slice(None, -step, None)
                value_padding = self.values.__getitem__(tuple(cut))
            elif fill_type == "matrix":
                value_padding = fillna
            else:
                raise ValueError(f"fill_type must be in roll, value, matrix, "
                                 f"the given input is {fill_type}")
            if list(value_padding.shape) != shape:
                raise ValueError(f"The given padding value's shape is not compitable, "
                                 f"padding value shape is {value_padding.shape}, "
                                 f"the XTensor shape is {shape}")
            value_padding = value_padding.swapaxes(0, axis)
            value_shifting = value_shifting.swapaxes(0, axis)
            shape = list(value_shifting.shape)
            shape[0] += value_padding.shape[0]
            new_values = np.empty(shape=shape)
            if step > 0:
                new_values[: value_padding.shape[0]] = value_padding
                new_values[value_padding.shape[0] :] = value_shifting
            else:
                new_values[: value_shifting.shape[0]] = value_shifting
                new_values[value_shifting.shape[0] :] = value_padding
            self.values = new_values.swapaxes(0, axis)
        else:
            obj = self.copy()
            obj.shift(step, axis, fill_type, True, fillna)
            return obj

    def swapaxes(self, axis1: int = 0, axis2: int = 1, inplace: bool = False):
        """swap the dimenstion of the data cube

        Args:
            axis1 (int): the first axis to be changed
            axis2 (int): the second axis to be changed
            inplace (bool):
        """
        if inplace:
            axis1 = axis1 % len(self.shape)
            axis2 = axis2 % len(self.shape)
            new_values = self._values.swapaxes(axis1, axis2)
            new_indexes = self.indexes.copy()
            if self.indexes:
                new_indexes[axis1], new_indexes[axis2] = self.indexes[axis2], self.indexes[axis1]
            self.assert_constructible(new_values, new_indexes)
            self._values, self._indexes = new_values, new_indexes
        else:
            obj = self.copy()
            obj.swapaxes(axis1, axis2, True)
            return obj

    def cumsum(self, axis: int = -1, inplace: bool = False):
        """Calculate cumulative sum on a given axis

        Args:
            axis (int=-1): The axis for the operation
            inplace (bool):
        """
        axis = axis % len(self.shape)
        if inplace:
            self.values = np.cumsum(self.values, axis)
        else:
            obj = self.copy()
            obj.cumsum(axis, True)
            return obj

    def cumprod(self, axis: int = -1, inplace: bool = False):
        """Calculate cumulative product on a given axis

        Args:
            axis (int=-1): The axis for the operation
            inplace (bool):
        """
        axis = axis % len(self.shape)
        if inplace:
            self.values = np.cumprod(self.values, axis)
        else:
            obj = self.copy()
            obj.cumprod(axis)
            return obj

    def diff(self, prepend: float = np.NaN, axis: int = -1, inplace: bool = False):
        """Calculate difference on certain axis

        Args:
            prepend (float): The value to fill for slots where the difference can't be calculated
            axis (int): The axis for the operation
            inplace (bool):
        """
        axis = axis % len(self.shape)
        if inplace:
            self.values = np.diff(self.values, prepend=prepend, axis=axis)
        else:
            obj = self.copy()
            obj.diff(prepend, axis, True)
            return obj

    def round(self, digits: int = 0, to: str = "", inplace: bool = False):
        """Round the the values to the specific digits

        Args:
            digits (int): The number of digits to be rounded
            to (str): Can be floor/ceil/"", deciding the round direction
        """
        if to and to not in {"floor", "ceil"}:
            raise ValueError(f"Parameter to must be in {{\"floor\", \"ceil\", None}}, "
                             f"given {to}")
        if inplace:
            if to == "ceil":
                self.values = self.values * (10 ** digits)
                self.values = np.round(np.ceil(self.values) / (10 ** digits), digits)
            elif to == "floor":
                self.values = self.values * (10 ** digits)
                self.values = np.round(np.floor(self.values) / (10 ** digits), digits)
            else:
                self.values = np.round(self.values, digits)
        else:
            obj = self.copy()
            obj.round(digits, to, True)
            return obj

    def copy(self):
        """copy a Data Cube

        Returns:
            XTensor: Deep copyed XTensor
        """
        res = XTensor(self._values.copy(), deepcopy(self._indexes))
        return res

    def to_dict(self) -> Dict[str, Iterable]:
        """Turn the data cube into a dict based on the index

        Returns:
            Dict[str, Iterable]: 转化后的结果
        """
        ret = dict()
        for key in self.indexes[0]:
            ret[key] = self[key]
        return ret

    def to_df(self) -> pd.DataFrame:
        """Transform Data Cube to be a dataframe. The Data Cube can only have two dimensions.

        Returns:
            pd.DataFrame: the transformed XTensor
        """
        if len(self.shape) != 2:
            raise ValueError("to_df fail: The dimensions number is not 2")
        return pd.DataFrame(self._values, index=self.indexes[0], columns=self.indexes[-1])

    def to_bytes(self) -> bytes:
        """Serialize XTensor into bytes

        Returns:
            bytes: A serialized XTensor, can be desearlized by from_bytes
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
    def from_bytes(cls, binary: bytes):
        """Deserialize XTensor from bytes

        Args:
            binary (bytes): the binary data
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
        self.assert_constructible(self._values, indexes)
        self._indexes = tuple([np.array(index) for index in enumerate(indexes)])

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values: object):
        self.assert_constructible(values, self.indexes)
        self._values = np.array(values)

    @property
    def T(self):
        """change the first dimension with the last dimension"""
        return self.swapaxes(0, -1)

    def astype(self, dtype: type):
        """change the dtype of data"""
        res = XTensor(self._values.astype(dtype), self._indexes)
        return res

    def __repr__(self):
        return (f"XTensor with shape:\n{self.shape}\n"
                f"indexes:\n{self.indexes}\nvalues:\n{self.values}")

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

    def _assert_compatibility(self, other):
        """Check the format of self and other XTensor so that
        they're compatible for arithmetic operation"""
        if isinstance(other, np.ndarray):
            other = XTensor(other)
            if other.shape == self.shape:
                return other
        elif isinstance(other, int) or isinstance(other, float):
            return XTensor(np.ones(shape=self.shape) * other)
        if not isinstance(other, XTensor):
            raise ValueError("Can't construct a XTensor based on the input")
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
        other = XTensor(self._assert_compatibility(other))
        values = self.values + other.values
        return XTensor(values, self._indexes)

    def __radd__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = other.values + self.values
        return XTensor(values, self._indexes)

    def __sub__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = self.values - other.values
        return XTensor(values, self._indexes)

    def __rsub__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = other.values - self.values
        return XTensor(values, self._indexes)

    def __mul__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = self.values * other.values
        return XTensor(values, self._indexes)

    def __rmul__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = other.values * self.values
        return XTensor(values, self._indexes)

    def __truediv__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = self.values / other.values
        return XTensor(values, self._indexes)

    def __rtruediv__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = other.values / self.values
        return XTensor(values, self._indexes)

    def __gt__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = self.values > other.values
        return XTensor(values, self._indexes)

    def __lt__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = self.values < other.values
        return XTensor(values, self._indexes)

    def __ge__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = self.values >= other.values
        return XTensor(values, self._indexes)

    def __le__(self, other):
        other = XTensor(self._assert_compatibility(other))
        values = self.values <= other.values
        return XTensor(values, self._indexes)

    def __invert__(self):
        return XTensor(~self.values, self._indexes)

    def __abs__(self):
        return XTensor(abs(self.values), self._indexes)
