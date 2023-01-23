
# Copyright (c) 2023 Apex Resource Management Solution Ltd. (ApexRMS). All rights reserved.
# MIT License

from setuptools import setup, find_packages
from pathlib import Path
from popfinder._version import __version__

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

entry_points = \
{'console_scripts': ['pop_classifier = popfinder.cli_classifier:main',
                     'pop_regressor = popfinder.cli_regressor:main']}

setup(name="popfinder",
      version=__version__,
      description="Genetic population assignment using neural networks",
      long_description=long_description,
      long_description_content_type='text/markdown',
      author="Katie Birchard",
      author_email="katie.birchard@apexrms.com",
      url=None,
      packages=find_packages(exclude="tests"),
      # install_requires=install_requires,
      entry_points=entry_points)