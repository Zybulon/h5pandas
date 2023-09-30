# h5pandas
h5pandas is a library that converts HDF5 Dataset from h5py into Pandas DataFrame.

### Readthedoc
Full API documentation is available at  : https://h5pandas.readthedocs.io/en/latest/

### Installation
```
pip install h5pandas
```

### Get started
How transform a h5py dataset to Pandas Dataframe:

```Python
"""Example of how to use h5pandas."""

import h5py
import h5pandas

arr = [[0.77129439, 0.68873990, 0.58298317, 0.38852130, 0.76915693],
       [0.25705227, 0.25732753, 0.23350236, 0.72443825, 0.82510932],
       [0.82022569, 0.60130446, 0.64930291, 0.53996334, 0.74156596],
       [0.47082073, 0.26073402, 0.99410667, 0.50356161, 0.49958255],
       [0.48200240, 0.68350121, 0.75641487, 0.00858738, 0.86024344],
       [0.18860991, 0.50119593, 0.20673441, 0.29877018, 0.92360508],
       [0.83575947, 0.89673302, 0.75841862, 0.70900089, 0.76026179],
       [0.68208926, 0.37177053, 0.83115045, 0.35738034, 0.47319340]]

# let's create a HDF5 file
with h5py.File("foo.h5", "w", libver='latest', driver="core") as f:
    d = f.create_dataset('bar', data=arr)

    # dataset_to_dataframe converts a dataset to a Pandas DataFrame
    df = h5pandas.dataset_to_dataframe(d, ["a", "b", "c", "d", "e"])
    # df is a Pandas DataFrame
    type(d)
```
> pandas.core.frame.DataFrame
