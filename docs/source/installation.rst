************
Installation
************

Requirements
############

Disimpy supports Python 3.11–3.14. GPU execution targets CUDA 12 or 13 and
NVIDIA GPUs with compute capability 7.5 or higher. Install an NVIDIA driver
compatible with the chosen CUDA version. Check your GPU on `NVIDIA's GPU list
<https://developer.nvidia.com/cuda-gpus>`_ and driver requirements in the
`CUDA release notes <https://developer.nvidia.com/cuda-toolkit-archive>`_.
If the CUDA Toolkit is not installed system-wide, install one set of toolkit
components with ``python -m pip install 'numba-cuda[cu12]'`` or use ``cu13``
for CUDA 13.

The current simulation backend is Numba-CUDA. Its `CUDA simulator
<https://nvidia.github.io/numba-cuda/user/simulator.html>`_ runs small tests
on a CPU without a GPU. It does not validate GPU compilation or performance.

Install the package
###################

Use a virtual environment. Install a release with pip:

.. code-block:: console

   python -m pip install disimpy

For development from a source checkout, create the portable Conda environment
and install the test dependencies:

.. code-block:: console

   conda env create -f development_environment.yml
   conda activate disimpy
   python -m pip install -e '.[test]'

Alternatively, install directly from GitHub:

.. code-block:: console

   python -m pip install git+https://github.com/kerkelae/disimpy.git

Run tests
#########

From the source checkout, run the small CPU and CUDA simulator tests without a
GPU:

.. code-block:: console

   NUMBA_ENABLE_CUDASIM=1 python -m pytest -q -m 'not gpu'

Set ``NUMBA_ENABLE_CUDASIM`` before Python imports Numba. The full numerical
suite requires a supported physical GPU. Run it with the simulator disabled:

.. code-block:: console

   python -m pytest -q

The installed package also exposes ``disimpy.tests.test_all()`` for the full
suite when the ``test`` extra is installed.
