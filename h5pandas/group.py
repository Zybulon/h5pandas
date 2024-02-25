"""Very thin overlay over h5py library."""

import numpy as np
from h5pandas.dataframe import dataset_to_dataframe, group_to_dataframe
from h5pandas import HDF5Dtype
import h5py

try:
    from pandas import DataFrame, Index
except ModuleNotFoundError:
    DataFrame = type(None)
    Index = type(None)


class Group(h5py.Group):
    """
    h5py Group that provides a DataFrame instead of dataset.

    See h5py documentation: https://docs.h5py.org/en/stable/high/group.html
    """

    def __init__(self, group_id, columns=None):
        """
        Transform an object into a Group that provides DataFrame instead of dataset.

        Parameters
        ----------
        group_id : TYPE
            DESCRIPTION.
        columns : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        """"""
        if isinstance(group_id, h5py.File):
            id = group_id["/"]._id
        elif isinstance(group_id, (h5py.h5f.FileID, h5py.h5g.GroupID)):
            id = group_id
        elif isinstance(group_id, h5py.Group):
            id = group_id._id

        super().__init__(id)

    def __getitem__(self, *args, **kwargs):
        """Convert item into DataFrame before returning it."""
        item = super().__getitem__(*args, **kwargs)
        if isinstance(item, h5py.Group):
            try:
                return group_to_dataframe(item)
            except Exception:
                return Group(item)
        elif isinstance(item, h5py.Dataset):
            try:
                return dataset_to_dataframe(item)
            except Exception:
                return item
        print("Item type not managed")
        return item

    def __getattribute__(self, name):
        """Get DataFrame if possible."""
        item = super().__getattribute__(name)
        if isinstance(item, h5py.File):
            item.__class__ = File
        elif isinstance(item, h5py.Group):
            item = Group(item)
        return item

    def create_dataset(
        self,
        name,
        shape=None,
        dtype=None,
        data=None,
        index: list | None | Index = None,
        columns: list[str] | None = None,
        metadata: dict = {},
        **kwargs,
    ):
        """
        Create a dataset.

        If columns is provided or if data is a DataFrame,
        the columns names are written as attribute of the dataset.
        If data is a DataFrame, its attributes (data.attrs) are saved into the
        dataset attributes so that they can be retrieve later with h5pandas.
        If metadata is provided, it is written inside the dataset attributes.
        If metadata as the same key as data.attrs, metadata will be written in the file.

        See h5py documentation: https://docs.h5py.org/en/stable/high/dataset.html

        Parameters
        ----------
        name: str
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        shape
            Dataset shape.  Use "()" for scalar datasets.  Required if "data"
            isn't provided.
        dtype
            Numpy dtype or string.  If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
        data
            Provide data to initialize the dataset.  If used, you can omit
            shape and dtype arguments.
        index: list, None or `pandas.Index`, optional
            Default=None.
            If not None, index will be written inside the HDF5 file and can be retrieve later with h5pandas.
        columns: list, optional
            names of the columns of the array to save, if any.
            If the array is a structured array and columns is none then structured names are used.
            Otherwise, if None, then nothing is written.
        metadata : dict, optional
            Additional metadata to save with the dataset attributes.


        Keyword-only arguments:

        Returns
        -------
        pandas.DataFrame
            The newly create DataFrame.

        """

        def add_attribute(dataset, attr_name, values):
            values = np.array(values)
            if values.dtype == object:
                values = values.astype(type(values[0]))
            try:
                dataset.attrs[attr_name] = np.char.encode(values)
            except TypeError:
                dataset.attrs[attr_name] = values

        # preprocess of numpy structured arrays
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = data.dtype.names
            if data.dtype.names is not None:
                # on destructure le numpy struct array si besoin
                from numpy.lib import recfunctions as rfn

                data = rfn.structured_to_unstructured(data)
        elif isinstance(data, DataFrame):
            # we look for properties inside the dataFrame
            metadata = data.attrs | metadata
            if columns is None:
                columns = list(data.columns)
            if index is None:
                index = data.index
            # In some cases we need to convert the dataframe
            # FIXME : H5Dtype need to inherit from pandas.NumpyDtype ?
            if isinstance(data.dtypes.iloc[0], HDF5Dtype):
                data = data.to_numpy(copy=False, dtype=data.dtypes.iloc[0].type)

        dataset = super().create_dataset(
            name, shape=shape, dtype=dtype, data=data, **kwargs
        )
        # Write columns name inside the dataFrame
        if columns is not None:
            add_attribute(dataset, "columns", columns)

        # Write index inside the dataFrame
        if index is not None:
            add_attribute(dataset, "index", index)

        # Write attributes inside the dataFrame
        for name, value in metadata.items():
            try:
                add_attribute(dataset, name, value)
            except Exception:
                print("Could not add {} metadata to h5file metadata".format(name))

        return dataset_to_dataframe(dataset, index=index, columns=columns)


class File(h5py.File, Group):
    """
    h5py File that provides a DataFrame instead of dataset.

    See h5py documentation: https://docs.h5py.org/en/stable/high/file.html
    """

    def __getitem__(self, *args, **kwargs):
        """Getter of Group class."""
        return super(h5py.File, self).__getitem__(*args, **kwargs)
