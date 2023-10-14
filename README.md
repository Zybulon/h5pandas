# h5pandas
h5pandas is thin overlay on top of h5py that provides pandas DataFrames instead of datasets.

This library has been developed to combine the power of Pandas' data manipulation with the efficiency of h5py and the HDF5 format.
Pandas already has a support for HDF5 files but it requires to load all the data at
file opening which can be very time and memory consuming for large dataset/DataFrame.

Instead, this library opens a file and create a Pandas DataFrame without actually reading the data.
The data are read only when needed, this behavior is called "lazy DataFrame".
It allows to open very huge files (even larger than memory) instantaneously and with a very memory footprint (less than 1 MB).

The particularity of these DataFrames is that they deeply rely on the underlying file object
Each time a data is accessed it reads it from this file object.
That means you need to make sure the underlying file is never closed.

When to use h5pandas : 
- When you have huge datasets and only want to read a few columns

When not to use h5pandas : 
- When you actually want to operate all the data

### Readthedoc
Full API documentation is available at  : https://h5pandas.readthedocs.io/en/latest/

### Installation
```
pip install h5pandas
```

### Get started

You can use h5pandas the same way you use h5py except that it works with DataFrame.
The main difference is that h5pandas gives pandas DataFrame objects instead of dataset objects, even if the file has not been written with h5pandas.

h5pandas is fully compatible with h5py: you can write a dataset with one library and read with the other indifferently.
The only difference is that h5pandas provides DataFrames instead of datasets objects.
```Python
import h5py
import h5pandas
import pandas as pd

df0 = pd.DataFrame([[0.09, 0.91, 0.23, 0.01, 0.02, 0.06],
                    [0.85, 0.67, 0.17, 0.25, 0.19, 0.11],
                    [0.92, 0.14, 0.52, 0.50, 0.43, 0.26],
                    [0.47, 0.47, 0.48, 0.72, 0.71, 0.12],
                    [0.05, 0.60, 0.12, 0.19, 0.20, 0.69],
                    [0.08, 0.64, 0.31, 0.98, 0.63, 0.05],
                    [0.74, 0.93, 0.76, 0.54, 0.03, 0.07],
                    [0.79, 0.98, 0.51, 0.73, 0.13, 0.31]],
                   columns=["f", "o", "o", "b", "a", "r"])

# you can write a DataFrame into a HDF5 file with create_dataset
# Inside the file, the columns names are saved as attribute of the dataset.
with h5pandas.File("foo.h5", "w") as file:
    df = file.create_dataset('bar', data=df0)

# Later you can retrieve your dataFrame with exactly the same columns names (instead of having datasets with h5py).
# If the data was not written with h5pandas, you will have a DataFrame but with no names
with h5pandas.File("foo.h5", "r") as file:
    df = file['/bar']

    # These DataFrames can operate with "classic" DataFrames
    delta = df - df0

    # you can still change columns names after DataFrame creation (it won't change them on the disk).
    df.columns = ["a", "b", "c", "d", "e", "g"]

    # With the "h5" accessor you can access to ...
    # The file
    df.h5.file
    # The dataset
    df.h5.dataset
    # The attributes
    df.h5.attrs


# If you already have a dataset from h5py you can convert it into a DataFrame with dataset_to_dataframe
with h5py.File("foo.h5", "r") as file:
    dataset = file['/bar']
    df = h5pandas.dataset_to_dataframe(dataset, ["a", "b", "c", "d", "e", "g"])

```
