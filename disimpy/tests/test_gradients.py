"""This module contains unit tests of the gradients module."""

import os
import numpy as np
import numpy.testing as npt

from .. import gradients


def load_example_gradient():
    T = 80e-3  # Duration of gradient array
    gradient_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'example_gradient.txt')
    gradient = np.loadtxt(gradient_file)[np.newaxis, :, :]
    dt = T / (gradient.shape[1] - 1)
    return gradient, dt


def test_interpolate_gradient():
    gradient_1, dt = load_example_gradient()
    gradient_2 = np.concatenate([gradient_1 for i in range(5)], axis=0)
    n_t = int(1e5)
    interp_g_1, dt_1 = gradients.interpolate_gradient(gradient_1, dt, n_t)
    interp_g_2, dt_2 = gradients.interpolate_gradient(gradient_2, dt, n_t)
    npt.assert_equal(interp_g_1.shape, (1, n_t, 3))
    npt.assert_equal(interp_g_2.shape, (5, n_t, 3))
    npt.assert_almost_equal(dt_1, dt * gradient_1.shape[1] / n_t, 10)
    npt.assert_almost_equal(dt_1, dt * gradient_2.shape[1] / n_t, 10)
    npt.assert_almost_equal(np.max(interp_g_1, axis=1),
                            np.max(gradient_1, axis=1))
    npt.assert_almost_equal(np.max(interp_g_2, axis=1),
                            np.max(gradient_2, axis=1))
    npt.assert_almost_equal(np.trapz(interp_g_1, axis=1, dx=dt_1),
                            np.trapz(gradient_1, axis=1, dx=dt))
    npt.assert_almost_equal(np.trapz(interp_g_2, axis=1, dx=dt_2),
                            np.trapz(gradient_2, axis=1, dx=dt))
    return


def test_calc_q():
    gradient_1, dt = load_example_gradient()
    gradient_2 = np.concatenate([gradient_1 for i in range(5)], axis=0)
    q_1 = gradients.calc_q(gradient_1, dt)
    q_2 = gradients.calc_q(gradient_2, dt)
    npt.assert_equal(q_1.shape, gradient_1.shape)
    npt.assert_equal(q_2.shape, gradient_2.shape)
    npt.assert_equal(q_1[:, -1, :], q_1[:, 0, :])
    npt.assert_equal(q_2[:, -1, :], q_2[:, 0, :])
    return


def test_calc_b():
    b = 3.19654721e+11  # Actual b-value
    gradient_1, dt = load_example_gradient()
    gradient_2 = np.concatenate([gradient_1 for i in range(5)], axis=0)
    b_1 = gradients.calc_b(gradient_1, dt)
    b_2 = gradients.calc_b(gradient_2, dt)
    npt.assert_almost_equal(b_1 / b, np.array([1]))
    npt.assert_almost_equal(b_2 / b, np.ones(5))
    return


def test_set_b():
    gradient_1, dt = load_example_gradient()
    gradient_2 = np.concatenate([gradient_1 for i in range(5)], axis=0)
    b = 1.25e9
    scaled_g_1 = gradients.set_b(gradient_1, dt, b)
    b_1 = gradients.calc_b(scaled_g_1, dt)
    scaled_g_2 = gradients.set_b(gradient_2, dt, b)
    b_2 = gradients.calc_b(scaled_g_2, dt)
    npt.assert_almost_equal(b_1 / b, np.array([1]))
    npt.assert_almost_equal(b_2 / b, np.ones(5))
    bs = np.arange(5) * 1e9
    npt.assert_raises(ValueError, gradients.set_b, gradient=gradient_2, dt=dt,
                      b=bs)
    bs = np.arange(1, 6) * 1e9
    scaled_g_2 = gradients.set_b(gradient_2, dt, bs)
    bs_2 = gradients.calc_b(scaled_g_2, dt)
    npt.assert_almost_equal(bs_2 / bs, np.ones(5))
    return


def test_rotate_gradient():
    gradient, _ = load_example_gradient()
    # a = np.array([1, 0, 0])  # Original direction
    b = np.array([0.20272312, 0.06456846, 0.97710504])  # Desired direction
    R = np.array([[0.20272312, -0.06456846, -0.97710504],
                  [0.06456846, 0.99653363, -0.0524561],
                  [0.97710504, -0.0524561, 0.20618949]])
    Rs = R[np.newaxis, :, :]
    rotated_g = gradients.rotate_gradient(gradient, Rs)
    d = rotated_g[0, 5, :]  # New gradient direction
    d /= np.linalg.norm(d)
    npt.assert_almost_equal(b, d)
    return
