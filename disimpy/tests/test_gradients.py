import os
import numpy as np
import numpy.testing as npt

from .. import gradients

def load_example_gradient():
    gradient_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'example_gradient.txt')
    gradient = np.loadtxt(gradient_file)[np.newaxis,:,:]
    return gradient
    
def test_interpolation():
    gradient_1 = load_example_gradient()
    gradient_2 = np.concatenate([load_example_gradient() for i in range(5)],
                                axis=0)
    dt = 80/(gradient_1.shape[1]-1)
    n_t = int(1e5)
    interp_g_1, dt_1 = gradients.interpolate_gradient(gradient_1, dt, n_t)
    interp_g_2, dt_2 = gradients.interpolate_gradient(gradient_2, dt, n_t)
    npt.assert_equal(interp_g_1.shape, (1, n_t, 3))
    npt.assert_equal(interp_g_2.shape, (5, n_t, 3))
    npt.assert_almost_equal(dt_1, gradient_1.shape[1]*dt/n_t, 10)
    npt.assert_almost_equal(dt_1, gradient_2.shape[1]*dt/n_t, 10)
    return

def test_b_value_calculation():
    return

def test_test_rotation():
    return