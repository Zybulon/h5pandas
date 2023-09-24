# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:10:22 2023

@author: Fred
"""

import numpy as np
from h5pandas import dataset_to_dataframe
from h5pandas import HDF5Dtype
import h5py
import time

HDF5Dtype("i8")


def TestGeneral():
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
        print('Test groupby')
        t0 = time.time()
        result = df.groupby("a", as_index=True)
        print(time.time()-t0)
        t0 = time.time()
        result.b.mean()
        print(time.time()-t0)
        t0 = time.time()

        # Tester opérations avec skipna

        # Tester hasna

        # Tester d'autres types

        # Test casting from a type to another

    # print(b)
    print('Total time : ', time.time() - t00)


if __name__ == '__main__':
    test_perso()
