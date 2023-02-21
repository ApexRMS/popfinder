Quickstart
==========
``popfinder`` is a Python package that uses neural networks for genetic population assignment. This tutorial will go through a basic workflow used to identify the most likely population of samples of unknown origin.
	
To complete this tutorial, you must `install popfinder`_.

	.. _install popfinder: https://popfinder.readthedocs.io/en/latest/install.html

Overview of ``popfinder``
-----------------------
``popfinder`` is a Python package designed to simplify the process of using classification and regression neural networks for genetic population assignment.

Set Up
------

Installing and Loading Python Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install ``popfinder`` using either ``conda install`` or ``pip install``. See the `Installation`_ page for more detailed installation instructions.

    .. _Installation: https://popfinder.readthedocs.io/en/latest/install.html

Then, in a new Python script, import the necessary modules from ``popfinder``.

.. code-block:: pycon

    >>> from popfinder.dataloader import GeneticData
    >>> from popfinder.classifier import PopClassifier
    >>> from popfinder.regressor import PopRegressor