"""Module related to dataFrames."""
import numpy as np
import pandas
from h5pandas.h5array import HDF5ExtensionArray, HDF5Dtype
import h5py


def dataframe_to_hdf5(
    dataframe: pandas.DataFrame,
    h5file: str | h5py.Group,
    dataset_name: str = "dataframe",
    metadata: dict = {},
    *args,
    **kwargs,
) -> h5py.Dataset:
    """
    Write a dataframe on into a HDF5 file.

    Parameters
    ----------
    dataframe : `pandas.DataFrame`
        The dataframe to write.
    h5file : str or `h5py.File` or `h5py.Group`
        If it is a string : the name of the HDF5 file in which the dataframe will be written.
        If the file already exist then the dataframe is added to this file. Otherwise the file is created.
        If hdf5file is a `h5py.File` or `h5py.Group` object then it will be written inside this object.
    dataset_name : str, optional
        The name of the dataset that will contain the dataframe. Default = "dataframe"
    metadata : dict, optional
        Additional metadata to save with the dataframe. Units or description for example.
    *args and **kwargs : additionnal parameters passed directly to h5py.create_dataset
        It can be compression options for example.
        See https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset
        and https://pypi.org/project/hdf5plugin/

    Returns
    -------
    dataset : `h5py.dataset`
        The dataset created inside h5file.
    """
    columns = dataframe.columns.values
    if isinstance(dataframe.dtypes.iloc[0], HDF5Dtype):
        dataframe = dataframe.to_numpy(copy=False, dtype=dataframe.dtypes.iloc[0].type)
    return _data_to_hf5(
        dataframe, h5file, columns, dataset_name, metadata, *args, **kwargs
    )


def ndarray_to_hdf5(
    array: np.ndarray,
    h5file: str | h5py.Group,
    columns: list[str] | None,
    dataset_name: str = "dataframe",
    metadata: dict = {},
    *args,
    **kwargs,
) -> h5py.Dataset:
    """
    Write a dataframe on into a HDF5 file.

    Parameters
    ----------
    array : `np.ndarray`
        The array to write.
    h5file : str or `h5py.File` or `h5py.Group`
        If it is a string : the name of the HDF5 file in which the dataframe will be written.
        If the file already exist then the dataframe is added to this file. Otherwise the file is created.
        If hdf5file is a `h5py.File` or `h5py.Group` object then it will be written inside this object.
    columns: list
        names of the columns of the array to save, if any.
        If the array is a structured array and columns is none then structured names are used.
        Otherwise, if None, then nothing is written.
    dataset_name : str, optional
        The name of the dataset that will contain the dataframe. Default = "dataframe"
    metadata : dict, optional
        Additional metadata to save with the dataframe. Units or description for example.
    *args and **kwargs : additionnal parameters passed directly to h5py.create_dataset
        It can be compression options for example.
        See https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset
        and https://pypi.org/project/hdf5plugin/

    Returns
    -------
    dataset : `h5py.dataset`
        The dataset created inside h5file.
    """
    from numpy.lib import recfunctions as rfn

    # preprocess of numpy arrays
    if isinstance(array, np.ndarray):
        if columns is None:
            columns = array.dtype.names
        if array.dtype.names is not None:
            # on destructure le numpy struct array si besoin
            array = rfn.structured_to_unstructured(array)
    return _data_to_hf5(array, h5file, columns, dataset_name, metadata, *args, **kwargs)


def _data_to_hf5(
    array,
    h5file: str | h5py.Group,
    columns: list[str] | None,
    dataset_name: str = "dataframe",
    metadata: dict = {},
    *args,
    **kwargs,
) -> h5py.Dataset:
    if isinstance(h5file, str):
        h5file = h5py.File(h5file, "a", libver=("v110", "latest"))
    elif not isinstance(h5file, h5py.Group):
        TypeError("h5file must be either a str, a h5py.Group or h5py.File.")

    if dataset_name in h5file:
        del h5file[dataset_name]

    dataset = h5file.create_dataset(
        dataset_name,
        data=array,
        chunks=(array.shape[0], 1),
        maxshape=[None] * len(array.shape),
        *args,
        **kwargs,
    )

    if columns is not None:
        dataset.attrs["columns"] = np.char.encode(np.array(columns).astype(str))

    for name, value in metadata.items():
        try:
            dataset.attrs[name] = np.char.encode(np.array(value))
        except TypeError:
            try:
                dataset.attrs[name] = value
            except Exception:
                print("Could not add {} metadata to h5file metadata".format(name))
    return dataset


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
            columns = tuple(np.char.decode(dataset.attrs["columns"]))
        else:
            columns = (None,) * dataset.shape[1]

    columns_decoded = [None] * len(columns)
    for i, col in enumerate(columns):
        if isinstance(col, (bytes, np.bytes_)):
            columns_decoded[i] = col.decode()
        else:
            columns_decoded[i] = col

    # we create a Series for each column
    series = (
        pandas.Series(HDF5ExtensionArray(dataset, i), name=col, copy=False)
        for i, col in enumerate(columns_decoded)
    )

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
        series.append(
            pandas.Series(HDF5ExtensionArray(dataset), name=dataset_name, copy=False)
        )

    # concatenate the series into a DataFrame
    return pandas.concat(series, axis=1)


def _group_fixed_to_dataframe(group) -> pandas.DataFrame:
    return dataset_to_dataframe(
        group["block0_values"], columns=group["axis0"], index=group["axis1"]
    )


def _group_table_to_dataframe(group) -> pandas.DataFrame:
    import warnings

    warnings.warn(
        "You should reconsider using h5pandas to open table dataset.", UserWarning
    )
    raise NotImplementedError(
        "You should reconsider using h5pandas to open table dataset."
    )


try:
    # delete the accessor to avoid warning
    del pandas.DataFrame.h5
    del pandas.Series.h5
except AttributeError:
    pass


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
        """Return the attributes of the dataset backing the Pandas Object."""
        return self._values._dataset.attrs

    @property
    def name(self):
        """Return the name of the dataset backing the Pandas Object."""
        return self._values._dataset.name
