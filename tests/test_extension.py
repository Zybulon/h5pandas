"""
Sandbox Tests.
"""

import numpy as np
import h5pandas
from h5pandas import dataset_to_dataframe
from h5pandas import HDF5Dtype
import h5py
import time
import pandas as pd

HDF5Dtype("i8")


def TestH5extensions():
    arr = np.random.rand(3000, 5)
    with h5py.File("toto2.h5", "w", libver='latest') as f:
        t0 = time.time()
        t00 = t0
        d = f.create_dataset('toto', data=arr)
        f.flush()
        t1 = time.time()
        # assert on time to make sure the file is not loaded
        print(d.dtype, t1-t0)
        t0 = time.time()

        df = dataset_to_dataframe(d)
        df = dataset_to_dataframe(d, ["a", "b", "c", "d", "e"])
        t1 = time.time()
        print(df, t1-t0)
        assert (t1-t0 < 0.02)
        d[0, 0] = 0
        assert (df.loc[0, 'a'] == 0)
        t0 = time.time()
        b = df["b"]

        ind = b < 0.5
        print(ind, time.time()-t0)
        t0 = time.time()

        b = df["b"]
        print(b)
        sub_df = df[ind]
        mini = df["a"].min()
        cosb = np.cos(df["b"])
        cosb_2 = np.cos(b)
        assert all(cosb == cosb_2)

        print('Test reduce std')
        res = b.std()

        print('Test accumulate cumsum')
        print(b.cumsum())

        # test groupby
        t0 = time.time()
        result = df.groupby("a", as_index=True)
        result.b.mean()
        print('Test groupby', time.time()-t0)

        t0 = time.time()
        result = df[(df['a'] < 0.5) & (df['b'] > 0.5)]
        result.c.mean()
        print('test loc :', time.time()-t0)

        print(df.h5.file)

        df["z"] = df["a"] + df["b"]*df["c"]
        df["y"] = df["a"] - df["b"]/df["c"]

        # Tester opérations avec skipna

        # Tester hasna

        # Tester d'autres types

        # Test casting from a type to another

    # print(b)
    print('Total time : ', time.time() - t00)


def TestH5Group():
    arr = np.random.rand(3000, 5)
    df = pd.DataFrame(arr)
    with h5py.File("toto3.h5", "w", libver='latest') as f:
        d = f.create_dataset('toto', data=df)

    with h5pandas.File("toto3.h5", "r", libver='latest') as f:
        df = f['toto']
        print(df)
        print(type(df))


if __name__ == '__main__':
    TestH5extensions()
    # TestH5Group()
