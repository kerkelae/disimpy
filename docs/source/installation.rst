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
The base package installs the small CUDA 12 runtime library required when the
CUDA backends are imported, including on machines running the CPU simulator.
If the CUDA Toolkit is not installed system-wide, install the package with one
set of toolkit components: ``python -m pip install 'disimpy[cuda12]'`` for CUDA
12 or ``python -m pip install 'disimpy[cuda13]'`` for CUDA 13.

GPU execution uses Numba-CUDA-MLIR. The test extra also installs legacy
Numba-CUDA so its `CUDA simulator
<https://nvidia.github.io/numba-cuda/user/simulator.html>`_ can run small tests
on a CPU. Simulator results do not validate MLIR compilation or GPU performance.

Install the package
###################

Use a virtual environment. Install a release with pip:

.. code-block:: console

   python -m pip install disimpy

For development and CPU simulator tests from a source checkout, create the
portable Conda environment and install the development requirements. This
installs Disimpy in editable mode, its test and documentation extras, and the
Ruff, build, and Twine tools. A CUDA Toolkit and NVIDIA driver are not needed
for these tests:

.. code-block:: console

   conda env create -f development_environment.yml
   conda activate disimpy
   python -m pip install -r requirements-dev.txt

For physical GPU simulations, install the appropriate CUDA Toolkit components
as well: ``python -m pip install -e '.[cuda13]'`` or use ``cuda12`` when
targeting CUDA 12.

Alternatively, install directly from GitHub:

.. code-block:: console

   python -m pip install git+https://github.com/kerkelae/disimpy.git

Run tests
#########

From the source checkout, run the small CPU and legacy CUDA simulator tests
without a GPU:

.. code-block:: console

   NUMBA_ENABLE_CUDASIM=1 python -m pytest -q -m 'not gpu'

Set ``NUMBA_ENABLE_CUDASIM`` before Python imports Disimpy. This selects the
legacy simulator backend. The full numerical suite requires a supported
physical GPU and the MLIR backend. Run it with the simulator disabled:

.. code-block:: console

   python -m pytest -q

The installed package also exposes ``disimpy.tests.test_all()`` for the full
suite when the ``test`` extra is installed.
