# Always prefer setuptools over distutils
from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path
from h5pandas import __version__

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="h5pandas",
    version=__version__,
    description="Load hdf5 into Pandas DataFrame instantaneously",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Frédéric MASSON",
    author_email="masson-frederic@hotmail.fr",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=["h5pandas"],
    include_package_data=True,
    install_requires=["h5py", "pandas>=2.1.0"],
)
