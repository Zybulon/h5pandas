"""Module related to dataFrames."""
import numpy as np
import pandas
from h5pandas.h5array import HDF5ExtensionArray
import h5py
import pandas


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

    columns_decoded = [None]*len(columns)
    for i, col in enumerate(columns):
        if isinstance(col, (bytes, np.bytes_)):
            columns_decoded[i] = col.decode()
        else:
            columns_decoded[i] = col

    # we create a Series for each column
    series = (pandas.Series(HDF5ExtensionArray(dataset, i), name=col, copy=False) for i, col in enumerate(columns_decoded))

    # concatenate the series into a DataFrame
    return pandas.concat(series, copy=copy, axis=1)


def group_to_dataframe(group) -> pandas.DataFrame:
    """
    Transform a group into a DataFrame.

    Parameters
    ----------
    group : h5py.group
        The group to convert into a DataFrame.

    Returns
    -------
    pandas.DataFrame
        A dataFrame backed by the dataset.
        If you change the dataset values, the DataFrame will cbe changed.
    """
    # First option : the dataframe has been written by pandas (PyTables) with format = "fixed" or "table"
    if "pandas_type" in group.attrs:
        if group.attrs["pandas_type"] == b"frame":
            return _group_fixed_to_dataframe(group)
        elif group.attrs["pandas_type"] == b"frame_table":
            return _group_table_to_dataframe(group)

    # Second option : all the datasets have the same length, each is one is a serie
    try:
        return _group_with_column_to_dataframe(group)
    except ValueError:
        pass

    raise ValueError("Group could not be converted into a DataFrame")


def _group_with_column_to_dataframe(group) -> pandas.DataFrame:
    series = []
    for dataset_name in group:
        dataset = group[dataset_name]
        if not isinstance(dataset, h5py.dataset):
            raise ValueError("All child of the group must be datasets")
        if "columns" in dataset.attrs:
            raise ValueError("This dataset contains several columns")
        series.append(pandas.Series(HDF5ExtensionArray(dataset), name=dataset_name, copy=False))

    # concatenate the series into a DataFrame
    return pandas.concat(series, axis=1)


def _group_fixed_to_dataframe(group) -> pandas.DataFrame:
    return dataset_to_dataframe(group["block0_values"], columns=group["axis0"], index=group["axis1"])


def _group_table_to_dataframe(group) -> pandas.DataFrame:
    raise NotImplementedError


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
