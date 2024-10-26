# Always prefer setuptools over distutils
from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the requirements from requirement.txt
try:
    with open(path.join(HERE, "requirements.txt"), encoding="utf-8") as f:
        requirements = f.readlines()
except FileNotFoundError:
    with open(
        path.join(HERE, "h5pandas.egg-info", "requires.txt"), encoding="utf-8"
    ) as f:
        requirements = f.readlines()

# Get the version number
with open(path.join(HERE, "h5pandas", "__init__.py"), encoding="utf-8") as f:
    lines = f.readlines()
version = lines[0][15:-2]

# This call to setup() does all the work
setup(
    name="h5pandas",
    version=version,
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
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=["h5pandas"],
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)
