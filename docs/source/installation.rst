************
Installation
************

Hardware requirements
#####################

You need an Nvidia's CUDA-capable GPU with compute capability 2.0 or above with
the appropriate Nvidia driver. You can check if your GPU is supported on
`Nvidia's website <https://developer.nvidia.com/cuda-gpus>`_.

Software requirements
#####################

You need the CUDA Toolkit version 8.0 or above. The CUDA Toolkit can be
installed from `Nvidia <https://developer.nvidia.com/cuda-toolkit>`_ or using
conda as described in `Numba's documentation
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html>`_. Make sure that
the CUDA Toolkit version you install supports your Nvidia driver or upgrade your
driver. If you are using the CUDA Toolkit not installed by conda and encounter
issues, check that the `installation path is configured so that Numba can access
it <https://numba.pydata.org/numba-doc/dev/cuda/overview.html#setting-cuda-installation-path>`_.

In addition, you need the following Python packages:

- matplotlib
- numba
- numpy
- pytest

These are automatically installed by pip if you follow the installation
instructions below.

Installation
############

To avoid possible issues with the dependencies, we recommend installing Disimpy
in a virtual environment or a conda environment. Disimpy can be installed with
pip by executing: ::

    pip install git+https://github.com/kerkelae/disimpy.git

Alternatively, given that the requirements specified above are met, you can use
Disimpy without installing by directly importing the functions you want to use
from the directory you can download
`here <https://github.com/kerkelae/disimpy/archive/master.zip>`_ or by cloning
the repository.

Automated tests
###############

To confirm that the installed simulator works, you should run the automated
tests by executing the following in the Python interpreter:

.. code-block:: python

   import disimpy.tests
   disimpy.tests.test_all()
