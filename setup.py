
# Copyright (c) 2023 Apex Resource Management Solution Ltd. (ApexRMS). All rights reserved.
# MIT License

from setuptools import setup, find_packages
from pathlib import Path
from popfinder._version import __version__

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name="popfinder",
      version=__version__,
      description="Genetic population assignment using neural networks",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author="Katie Birchard",
      author_email="katie.birchard@apexrms.com",
      url="https://apexrms.com/",
      packages=find_packages(exclude="tests"))

install_requires=['numpy', 'pandas', 'torch', 'scikit-learn', 'dill', 'seaborn',
                  'matplotlib', 'scikit-allel', 'zarr', 'h5py', 'scipy']

extras_require=['numpy', 'pandas', 'torch', 'scikit-learn', 'dill', 'seaborn',
                'matplotlib', 'scikit-allel', 'zarr', 'h5py', 'scipy']