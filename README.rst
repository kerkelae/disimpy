*******
Disimpy
*******

Disimpy is a tool for generating synthetic diffusion magnetic resonance imaging
data that is useful in model development and validation. Synthetic data is
generated following the Monte Carlo simulation framework established by
Hall et al. in "Convergence and Parameter Choice for Monte-Carlo Simulations of
Diffusion MRI" (DOI: 10.1109/TMI.2009.2015756). The simulations are executed on
Nvidia's CUDA-capable GPUs in a massively parallelized way, so using Disimpy
requires a CUDA-capable GPU and the CUDA toolkit.

Requirements
============

See more information `here 
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html>`_.

Installation
============

Disimpy can be installed by cloning this repository and installing with pip

    pip install .

Automated tests
===============

To confirm that the installation and CUDA configuration works, run the tests
by executing

>>> from disimpy import tests
>>> tests.test()
    
Usage example
=============

See notebook ``tutorial.ipynb`` to learn how to perform simulations.



References
==========

.. [camino] Hall, Matt G., and Daniel C. Alexander. "Convergence and parameter choice for Monte-Carlo simulations of diffusion MRI." IEEE transactions on medical imaging 28.9 (2009): 1354-1364.


