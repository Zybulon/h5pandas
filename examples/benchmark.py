"""Benchmark h5pandas vs pandas' HDF5 and pandas' feather."""

import numpy as np
import pandas as pd
import h5pandas as h5pd
import time
import hdf5plugin
import os
import gc
import tracemalloc
import matplotlib.pyplot as plt
import pyarrow
import hdf5plugin

file1 = "h5pandas.h5"
file2 = "pandas.h5"
file3 = "feather.feather"
nrows = 80000
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
print(f"    h5pandas.dataframe_to_hdf : {t1 - t0:.4}s for {st1 / (1024) ** 2:.6}MB")
print(f"    pandas.to_hdf (PyTables) : {t2 - t1:.4}s for {st2 / (1024) ** 2:.6}MB")

gc.collect()
tracemalloc.start()
t0 = time.time()
file = h5pd.File(file1, mode="r")
df1 = file["dataframe"]
a = df1[0] * df1[2] + df1[1]
t1 = time.time()
_, m1 = tracemalloc.get_traced_memory()
gc.collect()
tracemalloc.reset_peak()
t2 = time.time()
df2 = pd.read_hdf(file2, key="dataframe")
a = df2[0] * df2[2] + df2[1]
t3 = time.time()
_, m2 = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("\nOpening file with without compression :")
print(
    f"    h5pandas.dataframe_to_hdf : time = {t1 - t0:.4}s  RAM = {m1 / (1024) ** 2:.6}MB"
)
print(
    f"    pandas.to_hdf (PyTables) : time = {t3 - t2:.4}s  RAM = {m2 / (1024) ** 2:.6}MB"
)

file.close()
gc.collect()
os.remove(file1)
os.remove(file2)

# WTIH COMPRESSION
# writing
t0 = time.time()
h5pd.dataframe_to_hdf(
    df,
    file1,
    **hdf5plugin.Blosc(cname="blosclz", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
)
t1 = time.time()
df.to_hdf(file2, key="dataframe", complib="blosc:blosclz", complevel=5)
t2 = time.time()
df.to_feather(file3, compression="lz4", compression_level=5)
t3 = time.time()
del df
del arr

gc.collect()

write_time = [t1 - t0, t2 - t1, t3 - t2]
file_size = [
    os.stat(file1).st_size / 1024**2,
    os.stat(file2).st_size / 1024**2,
    os.stat(file3).st_size / 1024**2,
]

print("\nWriting time with compression level 5 :")
print(f"    h5pandas.dataframe_to_hdf : {t1 - t0:.4}s for {file_size[0]:.6}MB")
print(f"    pandas.to_hdf (PyTables) : {t2 - t1:.4}s for {file_size[1]:.6}MB")
print(f"    pandas.to_feather (pyarrow) : {t3 - t2:.4}s for {file_size[2]:.6}MB")

# Time evaluation
t0 = time.time()
file = h5pd.File(file1, mode="r")
df1 = file["dataframe"]
a = df1[0] * df1[2] + df1[1]
t1 = time.time()

t2 = time.time()
df2 = pd.read_hdf(file2, key="dataframe")
a = df2[0] * df2[2] + df2[1]
t3 = time.time()

t4 = time.time()
df3 = pd.read_feather(file3)
a = df3[0] * df3[2] + df3[1]
t5 = time.time()

read_time = [t1 - t0, t3 - t2, t5 - t4]

file.close()
del df1, df2, df3, a


# RAM evaluation
gc.collect()
tracemalloc.start()
file = h5pd.File(file1, mode="r")
df1 = file["dataframe"]
a = df1[0] * df1[2] + df1[1]
_, m1 = tracemalloc.get_traced_memory()
gc.collect()
tracemalloc.reset_peak()

df2 = pd.read_hdf(file2, key="dataframe")
a = df2[0] * df2[2] + df2[1]
_, m2 = tracemalloc.get_traced_memory()
gc.collect()
tracemalloc.reset_peak()

df3 = pd.read_feather(file3)
a = df3[0] * df3[2] + df3[1]
_, m3 = tracemalloc.get_traced_memory()

del df1, df2, df3, a
gc.collect()
tracemalloc.stop()

ram_usage = [m1 / 1024**2, m2 / 1024**2, m3 / 1024**2]
read_time = [t1 - t0, t3 - t2, t5 - t4]

print("\nOpening file with with compression :")
print(f"\th5pandas.File : time = {t1 - t0:.4}s  RAM = {ram_usage[0]:.6}MB")
print(f"\tpandas.read_hdf : time = {t3 - t2:.4}s  RAM = {ram_usage[1]:.6}MB")
print(f"\tpandas.read_feather : time = {t5 - t4:.4}s  RAM = {ram_usage[2]:.6}MB")

file.close()
gc.collect()

os.remove(file1)
os.remove(file2)
os.remove(file3)


# Plots
def bar(nplot: int, values, units: str, title: str):
    """Bar plot of performances."""
    libs = (
        "h5pandas\n(h5py)",
        "pandas HDF5\n(PyTables)",
        "pandas Feather\n(PyArrow)",
    )
    ax = plt.subplot(2, 2, nplot)

    bar = ax.bar(libs, values, width=0.4, color=["coral", "darkblue", "grey"])
    ax.set_ylabel(units)
    ax.set_title(title)
    labels = [f"{value:.1f}" for value in values]
    ax.bar_label(bar, labels)
    ax.set_ylim([0, 1.15 * ax.get_ylim()[1]])


plt.close("all")
fig = plt.figure(figsize=(8, 6))
bar(1, write_time, "(s)", "Writing time with\ncompression level=5")
bar(2, file_size, "(MB)", "File size")
bar(3, read_time, "(s)", "Time for opening the file\nand reading 3 columns")
bar(
    4,
    ram_usage,
    "(MB)",
    "RAM usage for opening the file\nand reading 3 columns",
)
plt.tight_layout(h_pad=2)
plt.savefig("performances.png")
