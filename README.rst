*******
Disimpy
*******

Disimpy is a tool for generating synthetic diffusion magnetic resonance imaging
data that is useful in the development and validation of new models and methods.
The data is generated according to the Monte Carlo simulation framework
established by Hall et al. [1]_.

Requirements
============

To use Disimpy, you need the following Python packages

- numba
- numpy
- matplotlib
- scipy

In addition, you will need an Nvidia's CUDA-enabled GPU with compute capability
2.0 or above with the appropriate Nvidia driver. Please see `Numba's documentation
<https://numba.pydata.org/numba-doc/dev/cuda/overview.html>`_. and `Nvidia's website 
<https://developer.nvidia.com/cuda-toolkit>`_ for more information.

Installation
============

Disimpy can be installed by cloning this repository and installing with pip

    pip install .

Automated tests
===============

To confirm that the installation and CUDA configuration works, you can run the
tests by executing the following in your Python console.

>>> from disimpy import tests
>>> tests.test()
    
Usage example
=============

Please see notebook `tutorial.ipynb
<https://github.com/kerkelae/disimpy/blob/master/tutorial.ipynb>`_ for usage
examples.

References
==========

.. [1] Hall, Matt G., and Daniel C. Alexander. "Convergence and parameter choice for Monte-Carlo simulations of diffusion MRI." IEEE transactions on medical imaging 28.9 (2009): 1354-1364. doi:`10.1109/TMI.2009.2015756 <https://ieeexplore.ieee.org/document/4797853>`_


