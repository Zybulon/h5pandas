# -*- coding: utf-8 -*-
"""Structured Dataset Class."""

import numpy as np
from h5pandas.h5array import dataset_to_dataframe
import h5py
try:
    from pandas import DataFrame
except ModuleNotFoundError:
    DataFrame = type(None)


class Group(h5py.Group):
    """Structured Group is a h5py group that provides StructuredDataset instead of Dataset when it can."""

    def __init__(self, group_id, columns=None):
        """Transform an object into a group that provides StructuredDataset when it can."""
        if isinstance(group_id, h5py.File):
            id = group_id['/']._id
        elif isinstance(group_id, (h5py.h5f.FileID, h5py.h5g.GroupID)):
            id = group_id
        elif isinstance(group_id, h5py.Group):
            id = group_id._id

        super().__init__(id)

    def __getitem__(self, *args, **kwargs):
        """Convert item into structured item before returning it."""
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
        """Get structured attribute if possible."""
        item = super().__getattribute__(name)
        if isinstance(item, h5py.File):
            item.__class__ = File
        elif isinstance(item, h5py.Group):
            item = Group(item)
        return item

    def create_dataset(self, name, shape=None, dtype=None, data=None, index=None, columns=None, **kwargs):
        """Get structured attribute if possible."""
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
    """Structured Group is a h5py group that provides StructuredDataset instead of Dataset when it can."""

    def __getitem__(self, *args, **kwargs):
        """Getter of Group class."""
        return super(h5py.File, self).__getitem__(*args, **kwargs)
