from . import test_utils, test_simulations, test_gradients

# Numpy test callable
from numpy.testing import Tester
test = Tester().test
del Tester