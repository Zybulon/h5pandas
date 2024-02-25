"""
Sandbox Tests.
"""
import os
import numpy as np
import h5pandas
from h5pandas import dataset_to_dataframe
from h5pandas import HDF5Dtype
from h5pandas import HDF5ExtensionArray
import h5py
import time
import pandas as pd
import gc

HDF5Dtype("i8")


def test_general_behavior():
    arr = np.random.rand(30000, 5)
    with h5py.File(
        "foobar.h5", "w", libver="latest", driver="core", backing_store=False
    ) as f:
        t0 = time.time()
        t00 = t0
        d = f.create_dataset("toto", data=arr)

        # very basic transformation in hdf5array
        array = HDF5ExtensionArray(d, 0)
        d[0, 0] = 1.0
        assert array[0] == 1.0

        # transformation into a series
        ser = pd.Series(array, name="toto", copy=False)
        ser.memory_usage()
        d[0, 0] = 2.0
        array[0] = 2.0
        assert ser[0] == 2.0
        assert d[0, 1] != 2.0

        t1 = time.time()
        print("Test opening file", t1 - t0)
        t0 = time.time()

        df = dataset_to_dataframe(d)
        df = dataset_to_dataframe(d, ["a", "b", "c", "d", "e"])
        t1 = time.time()
        print("Test conversion", t1 - t0)
        # assert on time to make sure the file is not loaded
        assert t1 - t0 < 0.02
        d[0, 0] = 0
        assert df.loc[0, "a"] == 0
        t0 = time.time()
        b = df["b"]

        ind = b < 0.5
        print("Test comparison", time.time() - t0)
        t0 = time.time()

        b = df["b"]
        df[ind]
        df["a"].min()
        cosb = np.cos(df["b"])
        cosb_2 = np.cos(b)

        assert all(cosb == cosb_2)
        cosb > cosb_2
        print("Test cos", time.time() - t0)
        t0 = time.time()

        b.std()
        print("Test std", time.time() - t0)
        t0 = time.time()

        b.cumsum()
        print("Test cumsum", time.time() - t0)
        t0 = time.time()

        # test groupby : bottleneck performances
        t0 = time.time()
        result = df.groupby("a", as_index=True)
        b = result.b
        b.mean()
        print("Test groupby", time.time() - t0)

        t0 = time.time()
        result = df[(df["a"] < 0.5) & (df["b"] > 0.5)]
        result.c.mean()
        print("Test loc :", time.time() - t0)
        t0 = time.time()

        df["z"] = df["a"] + df["b"] * df["c"]
        df["y"] = df["a"] - df["b"] / df["c"]
        df["z"] = df["a"] % df["b"]
        df["z"] = df["a"] % df["b"]
        print("Test basic operations", time.time() - t0)
        t0 = time.time()

        # Test operations with skipna

        # Test hasna

        # Test other types

        # Test casting from a type to another

    # print(b)
    print("Total time : ", time.time() - t00)


def test_rmul():
    arr = np.random.rand(3000, 5)
    with h5py.File(
        "foobar.h5", "w", libver="latest", driver="core", backing_store=False
    ) as f:
        d = f.create_dataset("toto", data=arr)
        df = dataset_to_dataframe(d)
        df[0] * 2.1


def test_op_2EA():
    arr = np.random.rand(3000, 5)
    with h5py.File(
        "foobar.h5", "w", libver="latest", driver="core", backing_store=False
    ) as f:
        d = f.create_dataset("toto", data=arr)
        df = dataset_to_dataframe(d)
        df[0] - df[0]


def test_add():
    arr = np.random.rand(3000, 5)
    with h5py.File(
        "foobar.h5", "w", libver="latest", driver="core", backing_store=False
    ) as f:
        d = f.create_dataset("toto", data=arr)
        df = dataset_to_dataframe(d)

        ser = df[0]
        ser + (-57.0)


def test_write_hdf5():
    arr = np.random.rand(3000, 5)
    with h5py.File(
        "foobar2.h5", "w", libver="latest", driver="core", backing_store=False
    ) as f:
        df0 = pd.DataFrame(arr)
        d = f.create_dataset("toto", data=arr)
        df = dataset_to_dataframe(d)
        df._values
        d2 = h5pandas.dataframe_to_hdf5(df, "foobar3.h5")
        df2 = dataset_to_dataframe(d2)
        assert (df2._values == df0._values).all()
    d2.file.close()
    os.remove("foobar3.h5")


def test_attributes():
    arr = np.random.rand(3000, 5)
    with h5py.File(
        "foobar2.h5", "w", libver="latest", driver="core", backing_store=False
    ) as f:
        df0 = pd.DataFrame(arr)
        df0.attrs = {"A": "A", "è": [1, 2, 3]}
        d = f.create_dataset("toto", data=arr)
        df = dataset_to_dataframe(d)
        df._values
        d2 = h5pandas.dataframe_to_hdf5(df, "foobar3.h5")
        df2 = dataset_to_dataframe(d2)
        assert (df2._values == df0._values).all()


def test_retrieve_dataframe():
    arr = np.random.rand(3000, 5)
    df = pd.DataFrame(arr)
    df_named = pd.DataFrame(
        arr, columns=["é", "b", "c", "d", "e"], index=range(1000, 4000)
    )

    df.to_hdf("foobar.h5", key="random_fixed", format="fixed", mode="w")
    df.to_hdf("foobar.h5", key="random_table", format="table")
    df_named.to_hdf("foobar.h5", key="named_random_fixed", format="fixed")
    df_named.to_hdf("foobar.h5", key="named_random_table", format="table")

    with h5pandas.File("foobar.h5", "a", libver="latest") as f:
        f.create_dataset("h5pandas", data=df)
        f.create_dataset("h5pandas_named", data=df_named)

    with h5pandas.File("foobar.h5", "r", libver="latest") as f:
        df_random_fixed = f["random_fixed"]
        assert (df._values == df_random_fixed._values).all()

        df_named_random_fixed = f["named_random_fixed"]
        assert all(df_named == df_named_random_fixed)

        df_h5pandas = f["h5pandas"]
        assert (df._values == df_h5pandas._values).all()

        df_h5pandas_named = f["h5pandas_named"]
        assert all(df_named == df_h5pandas_named)

    os.remove("foobar.h5")


def test_retrieve_index_and_columns_string():
    arr = np.random.rand(3000, 5)
    index = [f"index_{i}" for i in range(1000, 4000)]
    df_named = pd.DataFrame(arr, columns=["é", "b", "c", "d", "e"], index=index)

    df_named.to_hdf("foobar.h5", key="named_random_fixed", format="fixed")
    h5pandas.dataframe_to_hdf5(
        df_named, "foobar.h5", dataset_name="named_random_h5pandas"
    )

    with h5pandas.File("foobar.h5", "r", libver="latest") as f:
        df_named_random_fixed = f["named_random_fixed"]
        assert all(df_named == df_named_random_fixed)

        df_named_random_h5pandas = f["named_random_h5pandas"]
        assert all(df_named == df_named_random_h5pandas)

    gc.collect()
    os.remove("foobar.h5")


def test_retrieve_index_and_columns_int():
    arr = np.random.rand(3000, 5)
    df = pd.DataFrame(arr, columns=None, index=range(1000, 4000))

    df.to_hdf("foobar.h5", key="random_fixed", format="fixed")
    h5pandas.dataframe_to_hdf5(df, "foobar.h5", dataset_name="random_h5pandas")

    with h5pandas.File("foobar.h5", "r", libver="latest") as f:
        df_random_fixed = f["random_fixed"]
        assert (df == df_random_fixed).all().all()

        df_random_h5pandas = f["random_h5pandas"]
        assert (df == df_random_h5pandas).all().all()

    gc.collect()
    os.remove("foobar.h5")


def test_retrieve_attributes():
    arr = np.random.rand(3000, 5)
    index = [f"index_{i}" for i in range(1000, 4000)]
    df_named = pd.DataFrame(arr, columns=["é", "b", "c", "d", "e"], index=index)
    df_named.attrs = {
        "A": "B",
        "C": [1, 2, 3],
        "D": ["E", "F"],
        "G": np.array([1.2, 2.4]),
    }
    h5pandas.dataframe_to_hdf5(df_named, "foobar.h5", dataset_name="dataframe")

    with h5pandas.File("foobar.h5", "r", libver="latest") as f:
        df_retrieved = f["dataframe"]
        for key in df_named.attrs.keys():
            if len(df_named.attrs[key]) and not isinstance(df_named.attrs[key], str):
                assert all(df_named.attrs[key] == df_retrieved.attrs[key])
            else:
                assert df_named.attrs[key] == df_retrieved.attrs[key]
    gc.collect()
    os.remove("foobar.h5")


if __name__ == "__main__":
    # test_general_behavior()
    # test_retrieve_dataframe()
    test_retrieve_attributes()
    # test_retrieve_index_and_columns_string()
    # test_retrieve_index_and_columns_int()
    # test_rmul()
    # test_op_2EA()
    # test_add()
    # test_write_hdf5()
