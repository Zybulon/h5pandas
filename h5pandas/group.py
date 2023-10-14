"""Very thin overlay over h5py library."""

import numpy as np
from h5pandas.dataframe import dataset_to_dataframe
import h5py
try:
    from pandas import DataFrame
except ModuleNotFoundError:
    DataFrame = type(None)


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
            id = group_id['/']._id
        elif isinstance(group_id, (h5py.h5f.FileID, h5py.h5g.GroupID)):
            id = group_id
        elif isinstance(group_id, h5py.Group):
            id = group_id._id

        super().__init__(id)

    def __getitem__(self, *args, **kwargs):
        """Convert item into DataFrame before returning it."""
        item = super().__getitem__(*args, **kwargs)
        if isinstance(item, h5py.Group):
            return Group(item)
        elif isinstance(item, h5py.Dataset):
            try:
                return dataset_to_dataframe(item)
            except Exception:
                return item
        print("type d'item non géré")
        return item

    def __getattribute__(self, name):
        """Get DataFrame if possible."""
        item = super().__getattribute__(name)
        if isinstance(item, h5py.File):
            item.__class__ = File
        elif isinstance(item, h5py.Group):
            item = Group(item)
        return item

    def create_dataset(self, name, shape=None, dtype=None, data=None, index=None, columns=None, **kwargs):
        """
        Create a DataFrame.

        If columns is provided or if data is a DataFrame, the columns names are written as attribute of the dataset.

        See h5py documentation: https://docs.h5py.org/en/stable/high/dataset.html

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        shape : TYPE, optional
            DESCRIPTION. The default is None.
        dtype : TYPE, optional
            DESCRIPTION. The default is None.
        data : TYPE, optional
            DESCRIPTION. The default is None.
        index : TYPE, optional
            DESCRIPTION. The default is None.
        columns : TYPE, optional
            DESCRIPTION. The default is None.
        kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        pandas.DataFrame
            The newly create DataFrame.

        """
        dataset = super().create_dataset(name, shape=shape, dtype=dtype, data=data, **kwargs)
        # we look for columns name inside the dataFrame
        if isinstance(data, DataFrame) and columns is None:
            columns = list(data.columns)
        # Write columns name inside the dataFrame
        # TODO : decode it with :
        if columns is not None:
            dataset.attrs["columns"] = np.char.encode(np.array(columns).astype(str))

        return dataset_to_dataframe(dataset, index=index, columns=columns)


class File(h5py.File, Group):
    """
    h5py File that provides a DataFrame instead of dataset.

    See h5py documentation: https://docs.h5py.org/en/stable/high/file.html
    """

    def __getitem__(self, *args, **kwargs):
        """Getter of Group class."""
        return super(h5py.File, self).__getitem__(*args, **kwargs)
