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

You need the following:

- Numba (version 3.7.0 or above)
- CUDA Toolkit (version 8.0 or above)

It is recommended to `install Numba using Conda
<https://numba.readthedocs.io/en/stable/user/installing.html>`_. CUDA Toolkit
can be installed `using Conda 
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html#software>`_ or from
`Nvidia <https://developer.nvidia.com/cuda-toolkit>`_. Make
sure that the version you install supports your Nvidia driver or upgrade the
driver. The driver requirements of each CUDA Toolkit version can be found in
the `release notes <https://developer.nvidia.com/cuda-toolkit-archive>`_.
If you use the CUDA Toolkit not installed by Conda and encounter issues, check
that the `installation path is configured so that Numba can access it
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html#setting-cuda-installation-path>`_.

Installation
############

It is strongly recommended to install Disimpy in a virtual environment. The
most recent release can be installed with pip:

.. code-block::

    pip install disimpy

For advanced users
******************

The package can be installed directly from source (note that you can specify
things such as branch and commit):

.. code-block::

    pip install git+https://github.com/kerkelae/disimpy.git

Automated tests
###############

To confirm that the installed simulator works, you should run tests by
executing the following in the Python interpreter:

.. code-block:: python

   from disimpy.tests import test_all
   test_all()
