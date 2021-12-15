************
Installation
************

Hardware requirements
#####################

You need an Nvidia CUDA-capable GPU with compute capability 2.0 or above with
the appropriate Nvidia driver. You can check if your GPU is supported on
`Nvidia's website <https://developer.nvidia.com/cuda-gpus>`_.

Software requirements
#####################

You need the CUDA Toolkit version 8.0 or above which can be installed from
`Nvidia <https://developer.nvidia.com/cuda-toolkit>`_ or using Conda as
described in `Numba's documentation 
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html#software>`_. Make
sure that the version you install supports your Nvidia driver or upgrade the
driver. The driver requirements of each CUDA Toolkit version can be found in
the `release notes <https://developer.nvidia.com/cuda-toolkit-archive>`_.
If you use the CUDA Toolkit not installed by Conda and encounter issues, check
that the `installation path is configured so that Numba can access it
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html#setting-cuda-installation-path>`_.

In addition, you need the following Python packages:

- ``matplotlib``
- ``numba``
- ``numpy``
- ``pytest``
- ``scipy``

These are automatically installed by pip if you follow the installation
instructions below.

Installation
############

To avoid possible dependency issues, it is recommended to install Disimpy in a
virtual environment.

The most recent release can be installed with pip: 

.. code-block::

    pip install disimpy

The latest code in the master branch with the most recent updates can be
installed with pip: 

.. code-block::

    pip install git+https://github.com/kerkelae/disimpy.git



Alternatively, given that the requirements are met, you can use Disimpy without
installing by downloading the latest code `here
<https://github.com/kerkelae/disimpy/archive/master.zip>`_ and directly
importing the functions you want to use from the downloaded modules.

Automated tests
###############

To confirm that the installed simulator works, you should run the automated
tests by executing the following in the Python interpreter:

.. code-block:: python

   import disimpy.tests
   disimpy.tests.test_all()
