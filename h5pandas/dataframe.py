"""Module related to dataFrames."""
import numpy as np
import pandas
from h5pandas.h5array import HDF5ExtensionArray


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
    series = (pandas.Series(HDF5ExtensionArray(dataset, i), name=col, copy=False) for i, col in enumerate(columns))

    # concatenate the series into a DataFrame
    return pandas.concat(series, copy=copy, axis=1)


@pandas.api.extensions.register_dataframe_accessor("h5")
@pandas.api.extensions.register_series_accessor("h5")
class DatasetAccessor:
    """Accessor to dataset for pandas object from h5pandas."""

    def __init__(self, pandas_obj):
        """
        Init the accessor of a Panda Series or DataFrame.

        Parameters
        ----------
        pandas_obj : pandas.Series or pandas.DataFrame

        """
        self._values = self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Verify the DataFrame is backed by opened h5file."""
        if isinstance(obj, pandas.DataFrame):
            values = obj[obj.columns[0]].values
        elif isinstance(obj, pandas.Series) and not hasattr(obj.values, "_datatset"):
            values = obj.values
        else:
            values = obj
        if hasattr(values, "_dataset"):
            return values
        else:
            raise AttributeError("Pandas Object must be backed by h5file.")

    @property
    def file(self):
        """Return the file backing the Pandas Object."""
        return self._values._dataset.file

    @property
    def dataset(self):
        """Return the dataset backing the Pandas Object."""
        return self._values._dataset

    @property
    def attrs(self):
        """Return the file backing the Pandas Object."""
        return self._values._dataset.attrs
