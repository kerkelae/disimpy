*******
Disimpy
*******

Disimpy is a simple tool for generating synthetic diffusion magnetic resonance
imaging data that may be useful in model development and validation. Synthetic
data is generated following the Monte Carlo simulation framework established by
Hall et al. in "Convergence and Parameter Choice for Monte-Carlo Simulations of
Diffusion MRI" (DOI: 10.1109/TMI.2009.2015756). The simulations are executed on
Nvidia's CUDA-capable GPUs in a massively parallelized way, so using Disimpy
requires a CUDA-capable GPU and the CUDA toolkit. See more information `here 
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html>`_.

Disimpy can be installed by cloning this repository and installing with pip

    pip install .

To confirm that the installation and CUDA configuration works, run the tests
by executing

>>> from disimpy import tests
>>> tests.test()
    

See notebook ``tutorial.ipynb`` to learn how to perform simulations.