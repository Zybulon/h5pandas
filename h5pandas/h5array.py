"""HDF5ExtensionArray module."""
import numpy as np
import h5py
import uuid
from h5pandas.h5datatype import HDF5Dtype
import numbers

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Sequence,
    cast,
)

import pandas

from pandas._libs import lib
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_scalar,
)

from pandas.core import (
    arraylike,
    nanops,
    missing,
    roperator,
)
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
    validate_indices,
)

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

from pandas.core.dtypes.missing import isna

from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    FillnaOptions,
    Iterator,
    NpDtype,
    PositionalIndexer,
    Scalar,
    SortKind,
    TakeIndexer,
    TimeAmbiguous,
    TimeNonexistent,
    npt,
)


def dataset_to_dataframe(dataset, columns=None, index=None, copy=False):
    """
    Transform a dataset into a DataFrame.

    Parameters
    ----------
    dataset : h5py.dataset
        The dataset to convert into a DataFrame.
    columns : iterable, optional
        Column labels to use for resulting frame when data does not have them,
        defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
        will perform column selection instead.
    index : Index or array-like, optional
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    copy : bool, optional
        Copy data from inputs.
        For dict data, the default of None behaves like ``copy=True``.  For DataFrame
        or 2d ndarray input, the default of None behaves like ``copy=False``.
        If data is a dict containing one or more Series (possibly of different dtypes),
        ``copy=False`` will ensure that these inputs are not copied.

    Returns
    -------
    pandas.DataFrame
        A dataFrame backed by the dataset.
        If you change the dataset values, the DataFrame will cbe changed.

    """
    """"""
    # if no columns we try to find columns or we construct a tuple of None
    if columns is None:
        if "columns" in dataset.attrs:
            columns = tuple(np.char.decode(dataset.attrs['columns']))
        else:
            columns = (None,)*dataset.shape[1]

    # we create a Series for each column
    series = (pandas.Series(HDF5ExtensionArray(dataset, i), name=col) for i, col in enumerate(columns))

    # concatenate the series into a DataFrame
    return pandas.concat(series, copy=copy, axis=1)


class HDF5ExtensionArray(pandas.core.arraylike.OpsMixin, pandas.api.extensions.ExtensionArray):

    _HANDLED_TYPES = (np.ndarray, numbers.Number, list)

    def __init__(self, dataset, column_index=0, indices=None, dtype=None) -> None:
        if not isinstance(dataset, (h5py.Dataset)):
            # if dtype is None:
            #     if isinstance(dataset, (list, tuple)):
            #         dataset = np.array()
            #     dtype = dataset.dtype
            if len(dataset):
                f = h5py.File("h5pyArray_{}".format(uuid.uuid4()), 'w', libver='latest', driver="core", backing_store=False)
                try:
                    dataset = f.create_dataset("column", data=dataset, shape=(len(dataset), 1), maxshape=(None, None), chunks=(len(dataset), 1), dtype=dtype)
                except OSError:
                    try:
                        dataset = f.create_dataset("column", data=dataset, shape=(len(dataset), 1), maxshape=(None, None), chunks=(len(dataset), 1), dtype=h5py.opaque_dtype(dtype))
                    except TypeError:
                        dataset = np.array(dataset, dtype=dtype)
                except TypeError:
                    dataset = np.array([dataset], dtype=dtype)
                    dataset = dataset.T

            else:
                dataset = np.empty((0, 1))
                column_index = 0
        self._dataset = dataset
        self._column_index = column_index
        self._dtype = HDF5Dtype(self._dataset.dtype)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: np.dtype | None = None, copy: bool = False):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type in this method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a np.dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        if dtype is None:
            return HDF5ExtensionArray(np.array(scalars, dtype=dtype))

        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype.type)
        return HDF5ExtensionArray(np.array(scalars, dtype=dtype))

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: np.dtype | None = None, copy: bool = False
    ):
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a np.dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        raise NotImplementedError(cls)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.
        ExtensionArray.factorize : Encode the extension array as an enumerated type.
        """

        return HDF5ExtensionArray(values)

    def __getitem__(self, item: PositionalIndexer):
        """
        Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.

            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None

            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

            * list[int]:  A list of int

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.

        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.

        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        item = check_array_indexer(self, item)
        if is_scalar(item):
            try:
                return self._dataset[item, self._column_index]
            except (ValueError, TypeError):
                raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                                 "(`None`) and integer or boolean arrays are valid indices")

        elif isinstance(item, slice) and item == slice(None):
            # We must return a shallow copy
            return HDF5ExtensionArray(self._dataset, self._column_index)
        elif isinstance(item, tuple) and (slice(None) in item or Ellipsis in item):
            return HDF5ExtensionArray(self._dataset[*item, self._column_index])
        else:
            # FIXME : AVOID COPY HERE
            return HDF5ExtensionArray(self._dataset[item, self._column_index])

    def __setitem__(self, key, value) -> None:
        """
        Set one or more values inplace.

        This method is not required to satisfy the pandas extension array
        interface.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : Extensionnp.dtype.type, Sequence[Extensionnp.dtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        # Some notes to the ExtensionArray implementor who may have ended up
        # here. While this method is not required for the interface, if you
        # *do* choose to implement __setitem__, then some semantics should be
        # observed:
        #
        # * Setting multiple values : ExtensionArrays should support setting
        #   multiple values at once, 'key' will be a sequence of integers and
        #  'value' will be a same-length sequence.
        #
        # * Broadcasting : For a sequence 'key' and a scalar 'value',
        #   each position in 'key' should be set to 'value'.
        #
        # * Coercion : Most users will expect basic coercion to work. For
        #   example, a string like '2018-01-01' is coerced to a datetime
        #   when setting on a datetime64ns array. In general, if the
        #   __init__ method coerces that value, then so should __setitem__
        # Note, also, that Series/DataFrame.where internally use __setitem__
        # on a copy of the data.
        # FIXME
        return self._dataset.__setitem__(key, value)

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._dataset)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over elements of the array."""
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self)):
            yield self[i]

    def isna(self) -> np.ndarray:
        """
        A 1-D array indicating if each value is missing.

        Returns
        -------
        numpy.ndarray or pandas.api.extensions.ExtensionArray
            In most cases, this should return a NumPy ndarray. For
            exceptional cases like ``SparseArray``, where returning
            an ndarray would be expensive, an ExtensionArray may be
            returned.

        Notes
        -----
        If returning an ExtensionArray, then

        * ``na_values._is_boolean`` should be True
        * `na_values` should implement :func:`ExtensionArray._reduce`
        * ``na_values.any`` and ``na_values.all`` should be implemented
        """
        return np.isnan(self._dataset[:, self._column_index])

    def __contains__(self, item: object) -> bool | np.bool_:
        """Return for `item in self`."""
        # GH37867
        # comparisons of any item to pd.NA always return pd.NA, so e.g. "a" in [pd.NA]
        # would raise a TypeError. The implementation below works around that.
        if is_scalar(item) and isna(item):
            if not self._can_hold_na:
                return False
            elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                return self._hasna
            else:
                return False
        else:
            # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
            # attribute "any"
            return (item == self._dataset[:, self._column_index]).any()  # type: ignore[union-attr]

    @ classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple instances of H5pyArray.

        Args:
            to_concat (list): List of H5pyArray instances to concatenate.

        Returns:
            H5pyArray: The concatenated H5pyArray.
        """
        f = h5py.File("h5pyArray_concat{}".format(uuid.uuid4()), 'w', libver='latest', driver="core", backing_store=False, locking=False)
        # TODO : utiliser l'argument out ?

        array = np.concatenate([subdataset._dataset[()] for subdataset in to_concat], axis=0)
        dataset = f.create_dataset("extension", data=array, shape=(len(array), 1), maxshape=(None, None), chunks=(len(array), 1))

        return cls(dataset)

    # error: Signature of "__eq__" incompatible with supertype "object"
    def __eq__(self, other: Any) -> ArrayLike:  # type: ignore[override]
        """Return for `self == other` (element-wise equality)."""
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        if isinstance(other, (pandas.Series, pandas.Index, pandas.DataFrame)):
            return NotImplemented
        else:
            return self._dataset[:, self._column_index] == other

    # error: Signature of "__ne__" incompatible with supertype "object"
    def __ne__(self, other: Any) -> ArrayLike:  # type: ignore[override]
        """Return for `self != other` (element-wise in-equality)."""
        return ~(self == other)

    # error: Signature of "__ne__" incompatible with supertype "object"

    def __lt__(self, other: Any) -> ArrayLike:  # type: ignore[override]
        """Return for `self < other` (element-wise in-equality)."""
        if isinstance(other, (pandas.Series, pandas.Index, pandas.DataFrame)):
            return NotImplemented
        else:
            return self._dataset[:, self._column_index] < other

    def __gt__(self, other: Any) -> ArrayLike:  # type: ignore[override]
        """Return for `self < other` (element-wise in-equality)."""
        if isinstance(other, (pandas.Series, pandas.Index, pandas.DataFrame)):
            return NotImplemented
        else:
            return self._dataset[:, self._column_index] > other

    def __le__(self, other: Any) -> ArrayLike:  # type: ignore[override]
        """Return for `self < other` (element-wise in-equality)."""
        if isinstance(other, (pandas.Series, pandas.Index, pandas.DataFrame)):
            return NotImplemented
        else:
            return self._dataset[:, self._column_index] <= other

    def __ge__(self, other: Any) -> ArrayLike:  # type: ignore[override]
        """Return for `self < other` (element-wise in-equality)."""
        if isinstance(other, (pandas.Series, pandas.Index, pandas.DataFrame)):
            return NotImplemented
        else:
            return self._dataset[:, self._column_index] >= other

    def to_numpy(
        self,
        dtype: np.dtype | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert to a NumPy ndarray.

        This is similar to :meth:`numpy.asarray`, but may provide additional control
        over how the conversion is done.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

        Returns
        -------
        numpy.ndarray
        """
        # We do the copy anyway
        num_array = self._dataset[:, self._column_index]
        try:
            num_array = num_array.view(dtype)
        except TypeError:
            num_array = num_array.astype(dtype)

    # def __array__(self, *args, **kwargs):
    #     return self._dataset[:, self._column_index]

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @ property
    def dtype(self) -> HDF5Dtype:
        """An instance of 'np.dtype'."""
        return self._dtype

    @ property
    def shape(self):
        """Return a tuple of the array dimensions."""
        return (len(self),)

    @ property
    def size(self) -> int:
        """The number of elements in the array."""
        # error: Incompatible return value type (got "signedinteger[_64Bit]",
        # expected "int")  [return-value]
        return np.prod(self.shape)  # type: ignore[return-value]

    @ property
    def ndim(self) -> int:
        """Extension Arrays are only allowed to be 1-dimensional."""
        return 1

    @ property
    def nbytes(self) -> int:
        """The number of bytes needed to store this object in memory."""
        # If this is expensive to compute, return an approximate lower bound
        # on the number of bytes needed.
        return NotImplemented

    def astype(self, dtype, copy: bool = True) -> ArrayLike:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        np.ndarray or pandas.api.extensions.ExtensionArray
            An ExtensionArray if dtype is Extensionnp.dtype,
            Otherwise a NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, HDF5Dtype):
            dtype = dtype._numpy_dtype

        return HDF5ExtensionArray(self._dataset[:, self._column_index].astype(dtype), self._column_index, dtype=dtype)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.core.algorithms import take

        # is worth to try to select only a few indices ?
        data = self._dataset[:, self._column_index]

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        # fill value should always be translated from the scalar
        # type for the array, to the physical storage type for
        # the data, before passing to take.

        result = take(data, indices, fill_value=fill_value,
                      allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype)

    def copy(self):
        """
        Return a copy of the array.

        Returns
        -------
        ExtensionArray
        """
        return HDF5ExtensionArray(self._dataset[:, self._column_index])

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs):
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array

        Raises
        ------
        NotImplementedError : subclass does not define accumulations
        """
        if skipna:
            meth = getattr(self._dataset[:, self._column_index], name, None)
        else:
            array = self._dataset[:, self._column_index]
            meth = getattr(array[not np.isnan(array)], name, None)
        if meth is None:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{name}'"
            )
        return meth(**kwargs)

    def _reduce(self, name: str, *, skipna: bool = True, min_count=0, **kwargs):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
        if skipna:
            meth = getattr(self._dataset[:, self._column_index], name, None)
        else:
            array = self._dataset[:, self._column_index]
            meth = getattr(array[not np.isnan(array)], name, None)
        if meth is None:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{name}'"
            )
        return meth(**kwargs)

    # ------------------------------------------------------------------------
    # Ops

    def __invert__(self):
        return type(self)(~self._ndarray)

    def __neg__(self):
        return type(self)(-self._ndarray)

    def __pos__(self):
        return type(self)(+self._ndarray)

    def __abs__(self):
        return type(self)(abs(self._ndarray))
