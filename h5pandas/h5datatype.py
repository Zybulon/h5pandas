"""Pandas ExtensionDataType."""
from pandas.api.extensions import register_extension_dtype, ExtensionDtype
import numpy as np
from pandas._libs.hashtable import object_hash
from pandas.api.extensions import ExtensionArray
from typing import (
    Any,
    TypeVar,
)
from pandas._typing import (
    DtypeObj,
    Shape,
    type_t,
)

# from pandas.core.dtypes.dtypes import PandasDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

# To parameterize on same ExtensionDtype
ExtensionDtypeT = TypeVar("ExtensionDtypeT", bound="ExtensionDtype")


@register_extension_dtype
class HDF5Dtype(ExtensionDtype):
    """ Ok"""

    _metadata: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        """


        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        dtype = np.dtype(*args, **kwargs)
        self._numpy_dtype = dtype
        self._type = dtype.type
        self._name = dtype.name + "[HDF5]"
        self._metadata = ("name",)
        self._kind = dtype.kind

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        """
        Check whether 'other' is equal to self.

        By default, 'other' is considered equal if either

        * it's a string matching 'self.name'.
        * it's an instance of this type and all of the attributes
          in ``self._metadata`` are equal between `self` and `other`.

        Parameters
        ----------
        other : Any

        Returns
        -------
        bool
        """
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        if isinstance(other, type(self)):
            return all(
                getattr(self, attr) == getattr(other, attr) for attr in self._metadata
            )
        return False

    def __hash__(self) -> int:
        # for python>=3.10, different nan objects have different hashes
        # we need to avoid that and thus use hash function with old behavior
        return object_hash(tuple(getattr(self, attr) for attr in self._metadata))

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def na_value(self) -> object:
        """
        Default NA value to use for this type.

        This is used in e.g. ExtensionArray.take. This should be the
        user-facing "boxed" version of the NA value, not the physical NA value
        for storage.  e.g. for JSONArray, this is an empty dictionary.
        """
        return np.nan

    @property
    def type(self) -> type_t[Any]:
        """
        The scalar type for the array, e.g. ``int``

        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``, assuming
        that value is valid (not NA). NA values do not need to be
        instances of `type`.
        """
        return self._type

    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV'), default 'O'

        This should match the NumPy dtype used when the array is
        converted to an ndarray, which is probably 'O' for object if
        the extension type cannot be represented as a built-in NumPy
        type.

        See Also
        --------
        numpy.dtype.kind
        """
        return self._kind

    @property
    def name(self) -> str:
        """
        A string identifying the data type.

        Will be used for display in, e.g. ``Series.dtype``
        """
        return self._name

    @property
    def names(self) -> list[str] | None:
        """
        Ordered list of field names, or None if there are no fields.

        This is for compatibility with NumPy arrays, and may be removed in the
        future.
        """
        return None

    @classmethod
    def construct_array_type(cls) -> type_t[ExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from h5pandas import HDF5ExtensionArray
        return HDF5ExtensionArray

    def empty(self, shape: Shape) -> type_t[ExtensionArray]:
        """
        Construct an ExtensionArray of this dtype with the given shape.

        Analogous to numpy.empty.

        Parameters
        ----------
        shape : int or tuple[int]

        Returns
        -------
        ExtensionArray
        """
        cls = self.construct_array_type()
        return cls._empty(shape, dtype=self)

    @classmethod
    def construct_from_string(
        cls: type_t[ExtensionDtypeT], string: str
    ) -> ExtensionDtypeT:
        r"""
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[H]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\[(?P<arg_name>.+)\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
        import re
        pattern = re.compile(r"^(\w+)\[HDF5\]$")
        match = pattern.match(string)
        if match:
            print("MATCH", match, match[0], match[1])
            print(match.groupdict())
            return cls(match[1])
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Check if we match 'dtype'.

        Parameters
        ----------
        dtype : object
            The object to check.

        Returns
        -------
        bool

        Notes
        -----
        The default implementation is True if

        1. ``cls.construct_from_string(dtype)`` is an instance
           of ``cls``.
        2. ``dtype`` is an object and is an instance of ``cls``
        3. ``dtype`` has a ``dtype`` attribute, and any of the above
           conditions is true for ``dtype.dtype``.
        """
        dtype = getattr(dtype, "dtype", dtype)

        if isinstance(dtype, (ABCSeries, ABCIndex, ABCDataFrame, np.dtype)):
            # https://github.com/pandas-dev/pandas/issues/22960
            # avoid passing data to `construct_from_string`. This could
            # cause a FutureWarning from numpy about failing elementwise
            # comparison from, e.g., comparing DataFrame == 'category'.
            return False
        elif dtype is None:
            return False
        elif isinstance(dtype, cls):
            return True
        if isinstance(dtype, str):
            try:
                return cls.construct_from_string(dtype) is not None
            except TypeError:
                return False
        return False

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.

        By default ExtensionDtypes are assumed to be non-numeric.
        They'll be excluded from operations that exclude non-numeric
        columns, like (groupby) reductions, plotting, etc.
        """
        return np.issubdtype(self._numpy_dtype, np.number)

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.

        By default, ExtensionDtypes are assumed to be non-numeric.
        Setting this to True will affect the behavior of several places,
        e.g.

        * is_bool
        * boolean indexing

        Returns
        -------
        bool
        """
        return self._numpy_dtype == np.dtype('bool')

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        """
        Return the common dtype, if one exists.

        Used in `find_common_type` implementation. This is for example used
        to determine the resulting dtype in a concat operation.

        If no common dtype exists, return None (which gives the other dtypes
        the chance to determine a common dtype). If all dtypes in the list
        return None, then the common dtype will be "object" dtype (this means
        it is never needed to return "object" dtype from this method itself).

        Parameters
        ----------
        dtypes : list of dtypes
            The dtypes for which to determine a common dtype. This is a list
            of np.dtype or ExtensionDtype instances.

        Returns
        -------
        Common dtype (np.dtype or ExtensionDtype) or None
        """
        if len(set(dtypes)) == 1:
            # only itself
            return self
        else:
            return None

    @property
    def _can_hold_na(self) -> bool:
        """
        Can arrays of this dtype hold NA values?
        """
        return True
