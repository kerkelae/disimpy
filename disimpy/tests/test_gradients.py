"""This module contains tests of the gradients module."""

import numpy as np
import numpy.testing as npt

from .. import gradients, utils


def example_gradient():
    T = 80e-3
    gradient = np.zeros((1, 1000, 3))
    gradient[0, 1:201, 0] = 0.1
    gradient[0, -201:-1, 0] = -0.1
    dt = T / (gradient.shape[1] - 1)
    return gradient, dt


def test_interpolate_gradient():
    gradient, dt = example_gradient()
    n_t = int(1e5)
    interp_g, interp_dt = gradients.interpolate_gradient(gradient, dt, n_t)
    npt.assert_equal(interp_g.shape, (1, n_t, 3))
    npt.assert_almost_equal(interp_dt, dt * gradient.shape[1] / n_t)
    npt.assert_almost_equal(np.max(interp_g), np.max(gradient))
    npt.assert_almost_equal(np.min(interp_g), np.min(gradient))
    npt.assert_almost_equal(
        gradients.calc_b(interp_g, interp_dt) / gradients.calc_b(gradient, dt), 1
    )
    return


def test_calc_q():
    gradient, dt = example_gradient()
    q = gradients.calc_q(gradient, dt)
    npt.assert_equal(q.shape, gradient.shape)
    for i in np.linspace(0, gradient.shape[1] - 1, 10).astype(int):
        npt.assert_almost_equal(
            q[:, i - 1, :],
            gradients.GAMMA * np.trapz(gradient[:, 0:i, :], dx=dt, axis=1),
        )
    return


def test_calc_b():
    gradient, dt = example_gradient()
    b = gradients.calc_b(gradient, dt)
    npt.assert_almost_equal(b / 1.07507347e10, 1)
    return


def test_set_b():
    gradient, dt = example_gradient()
    gradient = np.concatenate([gradient for i in range(5)], axis=0)
    b = 1e9
    scaled_g = gradients.set_b(gradient, dt, b)
    npt.assert_equal(np.isclose(gradients.calc_b(scaled_g, dt), b), True)
    bs = np.arange(5) * 1e10
    scaled_g = gradients.set_b(gradient, dt, bs)
    npt.assert_equal(np.isclose(gradients.calc_b(scaled_g, dt), bs), True)
    npt.assert_raises(Exception, gradients.set_b, gradient=scaled_g, dt=dt, b=1e9)
    return


def test_rotate_gradient():
    gradient, _ = example_gradient()
    k = np.array([0.1, 0.5, -0.9])
    R = utils.vec2vec_rotmat(np.array([1, 0, 0]), k)
    Rs = R[np.newaxis, :, :]
    rotated_g = gradients.rotate_gradient(gradient, Rs)
    d = rotated_g[0, 5, :]
    npt.assert_almost_equal(k / np.linalg.norm(k), d / np.linalg.norm(d))
    Rs = np.ones((1, 3, 3))
    npt.assert_raises(ValueError, gradients.rotate_gradient, gradient=gradient, Rs=Rs)
    return


def test_pgse():
    delta = 15e-3
    DELTA = 50e-3
    bvals = np.array([1e9, 2e9, 3e9])
    bvecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    n_t = int(1e4)
    gradient, dt = gradients.pgse(delta, DELTA, n_t, bvals, bvecs)
    npt.assert_equal(gradient.shape, (len(bvals), n_t, 3))
    npt.assert_equal(np.all(gradient[:, 0, :] == 0), True)
    npt.assert_equal(np.all(gradient[:, -1, :] == 0), True)
    npt.assert_almost_equal(np.sum(gradient, axis=1), 0)
    for i in range(3):
        npt.assert_almost_equal(
            np.sum(
                (np.abs(gradient[i, 0 : int(n_t / 2), :]) > np.finfo(float).resolution)
            )
            * dt,
            delta,
            5,
        )
        npt.assert_almost_equal(
            np.sum(
                (np.abs(gradient[i, int(n_t / 2) : :, :]) > np.finfo(float).resolution)
            )
            * dt,
            delta,
            5,
        )
    npt.assert_almost_equal(gradients.calc_b(gradient, dt) / 1e9, bvals / 1e9)
    npt.assert_almost_equal(
        gradient[:, 1] / np.linalg.norm(gradient[:, 1], axis=1), bvecs
    )
    return
