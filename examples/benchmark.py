# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:42:08 2024

@author: Fred
"""
import numpy as np
import pandas as pd
import h5pandas as h5pd
import time
import hdf5plugin
import os
import gc
import tracemalloc

file1 = "h5pandas.h5"
file2 = "pandas.h5"
nrows = 100000
arr = np.random.rand(nrows, 3000)
arr[:, 0] = np.arange(nrows)
arr[:, 1] = np.zeros(nrows)
arr[:, 2] = np.ones(nrows)
df = pd.DataFrame(arr)

t0 = time.time()
h5pd.dataframe_to_hdf(df, file1)
t1 = time.time()
df.to_hdf(file2, key="dataframe", complevel=0)
t2 = time.time()
st1 = os.stat(file1).st_size
st2 = os.stat(file2).st_size

print("\nWriting time without compression :")
print(f"    h5pandas.dataframe_to_hdf : {t1-t0:.4}s for {st1/(1024)**2:.6}Mo")
print(f"    pandas.to_hdf (PyTables) : {t2-t1:.4}s for {st2/(1024)**2:.6}Mo")

gc.collect()
tracemalloc.start()
t0 = time.time()
file = h5pd.File(file1, mode="r")
df1 = file["dataframe"]
a = df1[0]*df1[2] + df1[1]
t1 = time.time()
_, m1 = tracemalloc.get_traced_memory()
gc.collect()
tracemalloc.reset_peak()
t2 = time.time()
df2 = pd.read_hdf(file2, key="dataframe")
a = df2[0]*df2[2] + df2[1]
t3 = time.time()
_, m2 = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("\nOpening file with without compression :")
print(f"    h5pandas.dataframe_to_hdf : time = {t1-t0:.4}s  RAM = {m1/(1024)**2:.6}Mo")
print(f"    pandas.to_hdf (PyTables) : time = {t3-t2:.4}s  RAM = {m2/(1024)**2:.6}Mo")

file.close()
gc.collect()
os.remove(file1)
os.remove(file2)

# WTIH COMPRESSION

t0 = time.time()
h5pd.dataframe_to_hdf(df, file1,  **hdf5plugin.Blosc(cname='blosclz', clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
t1 = time.time()
df.to_hdf(file2, key="dataframe", complib="blosc:blosclz", complevel=5)
t2 = time.time()
st1 = os.stat(file1).st_size
st2 = os.stat(file2).st_size

print("\nWriting time with blosclz compression level 5 :")
print(f"    h5pandas.dataframe_to_hdf : {t1-t0:.4}s for {st1/(1024)**2:.6}Mo")
print(f"    pandas.to_hdf (PyTables) : {t2-t1:.4}s for {st2/(1024)**2:.6}Mo")

gc.collect()
tracemalloc.start()
t0 = time.time()
file = h5pd.File(file1, mode="r")
df1 = file["dataframe"]
a = df1[0]*df1[2] + df1[1]
t1 = time.time()
_, m1 = tracemalloc.get_traced_memory()
gc.collect()
tracemalloc.reset_peak()
t2 = time.time()
df2 = pd.read_hdf(file2, key="dataframe")
a = df2[0]*df2[2] + df2[1]
t3 = time.time()
_, m2 = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("\nOpening file with without compression :")
print(f"    h5pandas.dataframe_to_hdf : time = {t1-t0:.4}s  RAM = {m1/(1024)**2:.6}Mo")
print(f"    pandas.to_hdf (PyTables) : time = {t3-t2:.4}s  RAM = {m2/(1024)**2:.6}Mo")

file.close()
gc.collect()
os.remove(file1)
os.remove(file2)
