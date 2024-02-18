Installation
============
``popfinder`` can be installed using either the ``conda`` or ``pip`` package managers. ``conda`` is a general package manager capable of installing packages from many sources, but ``pip`` is strictly a Python package manager. 


Dependencies
------------

``popfinder`` was tested and developed using **Python 3.10**. It has the following dependencies:

* numpy
* pandas
* torch
* scikit-learn
* dill
* seaborn
* matplotlib
* scikit-allel
* scipy

Using conda
-----------

Follow these steps to get started with ``conda`` and use ``conda`` to install ``popfinder``. 

1. Install ``conda`` using the Miniconda or Anaconda installer (in this tutorial we use Miniconda). To install Miniconda, follow `this link`_ and under the **Latest Miniconda Installer Links**, download Miniconda for your operating system. Open the Miniconda installer and follow the default steps to install ``conda``. For more information, see the `conda documentation`_.

	.. _this link: https://docs.conda.io/en/latest/miniconda.html
	.. _conda documentation: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

2. To use ``conda``, open the command prompt that was installed with the Miniconda installer. To find this prompt, type "anaconda prompt" in the **Windows Search Bar**. You should see an option appear called **Anaconda Prompt (miniconda3)**. Select this option to open a command line window. All code in the next steps will be typed in this window. 

3. You can either install ``popfinder`` and its dependencies into your base environment, or set up a new ``conda`` environment (recommended). Run the code below to set up and activate a new ``conda`` environment called "myenv" that uses Python 3.10.

.. code-block:: console

	# Create new conda environment
	conda create -n myenv python=3.10

	# Activate environment
	conda activate myenv

You should now see that "(base)" has been replaced with "(myenv)" at the beginning of each prompt.

4. Set the package channel for ``conda``. To be able to install ``popfinder``, you need to access the ``conda-forge`` package channel. To configure this channel, run the following code in the Anaconda Prompt.

.. code-block:: console

	# Set conda-forge package channel
	conda config --add channels conda-forge

5. Install ``popfinder`` using ``conda install``.

.. code-block:: console

	# Install popfinder
	conda install popfinder

	# Install pytorch from the pytorch channel
	conda install -c pytorch pytorch

``popfinder`` should now be installed and ready to use!

Using pip
---------

Use ``pip`` to install ``popfinder`` to your default python installation. You can install Python from `www.python.org`_. You can also find information on how to install ``pip`` from the `pip documentation`_.

	.. _www.python.org: https://www.python.org/downloads/
	.. _pip documentation: https://pip.pypa.io/en/stable/installation/

Install ``popfinder`` using ``pip install``. Make sure to also install any missing dependencies to your environment.

.. code-block:: console

	# Make sure you are using the latest version of pip
	pip install --upgrade pip

	# Install popfinder
	pip install popfinder

	# Install any dependencies that are missing
	pip install numpy pandas torch scikit-learn dill seaborn matplotlib scikit-allel zarr h5py scipy
