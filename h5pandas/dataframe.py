"""Module related to dataFrames."""
import numpy as np
import pandas
from h5pandas.h5array import HDF5ExtensionArray
import h5py


def dataframe_to_hdf5(
    dataframe: pandas.DataFrame,
    h5file: str | h5py.Group,
    dataset_name: str = "dataframe",
    index: list | None | pandas.Index = None,
    columns: list[str] | None = None,
    metadata: dict = {},
    *args,
    **kwargs,
) -> h5py.Dataset:
    """
    High-level function to write a DataFrame into a HDF5 file.

    Dataframe columns names (dataframe.columns) and attributes (dataframe.attrs)
    will be written inside the dataset attributes and can be retrieve later
    when accessing the file with h5pandas.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe to write.
    h5file : str or `h5py.File` or `h5py.Group`
        If it is a string : the name of the HDF5 file in which the dataframe will be written.
        If the file already exist then the dataframe is added to this file.
        Otherwise the file is created.
        If hdf5file is a `h5py.File` or `h5py.Group` object then it will be written inside this object.
    dataset_name : str, optional
        The name of the dataset that will contain the dataframe. Default = "dataframe".
    index: list, None or `pandas.Index`, optional
        Default=None.
        If not None, index will be written inside the HDF5 file and can be retrieve later with h5pandas.
    columns: list, optional
        names of the columns of the dataframe to save, if any.
        If columns is none then the dataframe names are used.
        Otherwise, if None, then nothing is written.
    metadata : dict, optional
        Additional metadata to save with the dataframe as dataset attributes. Units or description for example.
    *args and **kwargs : additionnal parameters passed directly to h5py.create_dataset
        It can be compression options for example.
        See https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset
        and https://pypi.org/project/hdf5plugin/

    Returns
    -------
    dataset : h5py.dataset
        The dataset created inside h5file.
    """
    return _data_to_hf5(
        dataframe,
        h5file=h5file,
        dataset_name=dataset_name,
        index=index,
        columns=columns,
        metadata=metadata,
        *args,
        **kwargs,
    )


def ndarray_to_hdf5(
    array: np.ndarray,
    h5file: str | h5py.Group,
    dataset_name: str = "array",
    index: list | None | pandas.Index = None,
    columns: list[str] | None = None,
    metadata: dict = {},
    *args,
    **kwargs,
) -> h5py.Dataset:
    """
    High-level function to write a NumpyArray into a HDF5 file.

    Parameters
    ----------
    array : np.ndarray
        The array to write.
    h5file : str or `h5py.File` or `h5py.Group`
        If it is a string : the name of the HDF5 file in which the array will be written.
        If the file already exist then the array is added to this file.
        Otherwise the file is created.
        If hdf5file is a `h5py.File` or `h5py.Group` object then it will be written inside this object.
    dataset_name : str, optional
        The name of the dataset that will contain the array. Default = "array".
    index: list, None or `pandas.Index`, optional
        Default=None.
        If not None, index will be written inside the HDF5 file and can be retrieve later with h5pandas.
    columns: list, optional
        names of the columns of the array to save, if any.
        If the array is a structured array and columns is none then structured names are used.
        Otherwise, if None, then nothing is written.
    metadata : dict, optional
        Additional metadata to save with the array as dataset attributes. Units or description for example.
    *args and **kwargs : additionnal parameters passed directly to h5py.create_dataset
        It can be compression options for example.
        See https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset
        and https://pypi.org/project/hdf5plugin/

    Returns
    -------
    dataset : h5py.dataset
        The dataset created inside h5file.
    """
    return _data_to_hf5(
        array,
        h5file=h5file,
        dataset_name=dataset_name,
        index=index,
        columns=columns,
        metadata=metadata,
        *args,
        **kwargs,
    )


def _data_to_hf5(
    array,
    h5file: str | h5py.Group,
    dataset_name: str = "dataframe",
    index: list | None | pandas.Index = None,
    columns: list[str] | None = None,
    metadata: dict = {},
    *args,
    **kwargs,
) -> h5py.Dataset:
    from h5pandas.group import File, Group

    if isinstance(h5file, str):
        h5file = File(h5file, "a", libver=("v110", "latest"))
    elif isinstance(h5file, h5py.Group) and not isinstance(h5file, Group):
        h5file = Group(h5file)
    elif not isinstance(h5file, h5py.Group):
        TypeError("h5file must be either a str, a h5py.Group or h5py.File.")

    if dataset_name in h5file:
        del h5file[dataset_name]

    # select default parameters for optimised dataframe writting
    if "chunks" not in kwargs:
        kwargs["chunks"] = (array.shape[0], 1)

    if "maxshape" not in kwargs:
        kwargs["maxshape"] = [None] * len(array.shape)

    dataframe = h5file.create_dataset(
        dataset_name,
        data=array,
        index=index,
        columns=columns,
        metadata=metadata,
        *args,
        **kwargs,
    )
    return dataframe.h5.dataset


def dataset_to_dataframe(dataset: h5py.Dataset, columns=None, index=None, copy=False):
    """
    Transform a dataset into a DataFrame.

    Parameters
    ----------
    dataset : h5py.Dataset
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
    # if no columns we try to find columns or we construct a tuple of None
    if columns is None:
        if "columns" in dataset.attrs:
            try:
                columns = tuple(np.char.decode(dataset.attrs["columns"]))
            except TypeError:
                columns = dataset.attrs["columns"]
        else:
            columns = (None,) * dataset.shape[1]

    columns_decoded = [None] * len(columns)
    for i, col in enumerate(columns):
        if isinstance(col, (bytes, np.bytes_)):
            columns_decoded[i] = col.decode()
        elif col is None:
            columns_decoded[i] = i
        else:
            columns_decoded[i] = col

    if index is None:
        if "index" in dataset.attrs:
            try:
                index = tuple(np.char.decode(dataset.attrs["index"]))
            except TypeError:
                index = dataset.attrs["index"]

    # we create a Series for each column
    series = (
        pandas.Series(HDF5ExtensionArray(dataset, i), index=index, name=col, copy=False)
        for i, col in enumerate(columns_decoded)
    )

    # concatenate the series into a DataFrame
    dataframe = pandas.concat(series, copy=copy, axis=1)
    # copy the dataset attrs into the dataframe attrs
    for key, value in dataset.attrs.items():
        if key in ("columns", "index"):
            continue
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode()
        try:
            value = np.char.decode(value)
        except (AttributeError, TypeError):
            pass
        dataframe.attrs[key] = value
    return dataframe


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
    if pandas.api.types.is_string_dtype(group["axis0"].dtype):
        columns = np.char.decode(group["axis0"])
    else:
        columns = group["axis0"]
    if pandas.api.types.is_string_dtype(group["axis1"].dtype):
        index = np.char.decode(group["axis1"])
    else:
        index = group["axis1"]
    return dataset_to_dataframe(group["block0_values"], columns=columns, index=index)


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
