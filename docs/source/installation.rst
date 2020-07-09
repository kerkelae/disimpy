************
Installation
************

Hardware requirements
#####################

You need an Nvidia's CUDA-enabled GPU with compute capability 2.0 or above with
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
We personally prefer not to use conda for installing the CUDA Toolkit.

In addition, you need the following Python packages:

- numba
- numpy
- scipy
- matplotlib
- pytest

These are automatically installed by pip if you follow the installation
instructions below.

Installation
############

To avoid possible issues with automatic installation of some of the dependencies
(llvmlite), we recommend installing Disimpy in a virtual environment or a conda
environment. Disimpy can be installed by downloading the source code from `here <https://github.com/kerkelae/disimpy/archive/master.zip>`_.
Then, extract the .zip file, navigate into the extracted directory, and install
with pip: ::

    pip install .

Alternatively, given that the requirements specified above are met, you can use
Disimpy without installing by directly importing the functions you want to use
from the extracted directory.

Automated tests
###############

To confirm that the installation and CUDA configuration works, you should run
the automated tests by executing the following in Python interpreter:

.. code-block:: python

   import disimpy.tests
   disimpy.tests.test_all()

Running the tests may take up to a few minutes depending on your GPU.
