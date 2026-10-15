"""This module contains tests of the gradients module."""

import os

import numpy as np
import numpy.testing as npt
import pytest

from .. import gradients, simulations, utils


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


@pytest.mark.parametrize(
    "gradient",
    [
        np.zeros((0, 2, 3)),
        np.zeros((1, 1, 3)),
        np.zeros((1, 2, 2)),
        np.full((1, 2, 3), np.nan),
        np.zeros((1, 2, 3), dtype=int),
    ],
)
def test_gradient_helpers_reject_invalid_arrays(gradient):
    with pytest.raises(ValueError, match="gradient"):
        gradients.calc_q(gradient, 1e-3)


@pytest.mark.parametrize("dt", [0.0, -1e-3, np.nan, np.inf, 1])
def test_gradient_helpers_reject_invalid_dt(dt):
    gradient, _ = example_gradient()
    with pytest.raises(ValueError, match="dt"):
        gradients.calc_q(gradient, dt)


def test_gradient_time_step_accepts_numpy_floating():
    gradient, _ = example_gradient()
    npt.assert_allclose(
        gradients.calc_q(gradient, np.float32(1e-3)),
        gradients.calc_q(gradient, float(np.float32(1e-3))),
    )


def test_simulation_accepts_same_numpy_floating_time_step():
    gradient = np.zeros((1, 2, 3))
    # The deliberately invalid substrate stops before any GPU setup. The
    # substrate error confirms that the NumPy float passed dt validation.
    with pytest.raises(ValueError, match="substrate"):
        simulations.simulation(1, 2e-9, gradient, np.float32(1e-3), None)


@pytest.mark.parametrize("n_t", [0, 1, 2.5])
def test_interpolate_rejects_invalid_sample_count(n_t):
    gradient, dt = example_gradient()
    with pytest.raises(ValueError, match="n_t"):
        gradients.interpolate_gradient(gradient, dt, n_t)


def test_calc_q():
    gradient, dt = example_gradient()
    q = gradients.calc_q(gradient, dt)
    npt.assert_equal(q.shape, gradient.shape)
    for i in np.linspace(0, gradient.shape[1] - 1, 10).astype(int):
        npt.assert_almost_equal(
            q[:, i - 1, :],
            gradients.GAMMA * np.trapezoid(gradient[:, 0:i, :], dx=dt, axis=1),
        )
    return


def test_calc_b():
    gradient, dt = example_gradient()
    b = gradients.calc_b(gradient, dt)
    npt.assert_almost_equal(b / 1.07507347e10, 1)
    return


def test_calc_b_uses_n_samples_over_n_minus_one_intervals():
    dt = 1e-3
    magnitude = 0.1
    gradient = np.zeros((1, 4, 3))
    gradient[0, :, 0] = [0, magnitude, -magnitude, 0]
    scale = gradients.GAMMA**2 * magnitude**2 * dt**3

    q = gradients.calc_q(gradient, dt)
    npt.assert_allclose(q[:, -1], 0, atol=1e-12)
    npt.assert_allclose(gradients.calc_b(gradient, dt), [11 / 20 * scale])


def test_nonrephased_b_excludes_terminal_position_encoding():
    dt = 2e-3
    magnitude = 0.2
    gradient = np.zeros((1, 2, 3))
    gradient[0, :, 0] = magnitude
    scale = gradients.GAMMA**2 * magnitude**2 * dt**3
    npt.assert_allclose(gradients.calc_b(gradient, dt), [scale / 3])


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


def test_set_b_zero_and_invalid_targets():
    gradient, dt = example_gradient()
    two_measurements = np.repeat(gradient, 2, axis=0)
    scaled = gradients.set_b(two_measurements, dt, [0, 1e9])
    npt.assert_array_equal(scaled[0], 0)
    npt.assert_allclose(gradients.calc_b(scaled, dt), [0, 1e9], rtol=1e-12)
    npt.assert_allclose(gradients.calc_b(gradients.set_b(gradient, dt, 2e9), dt), [2e9])
    npt.assert_array_equal(gradients.set_b(np.zeros_like(gradient), dt, 0), 0)

    for invalid in (-1, np.nan, np.inf, [1e9, np.nan]):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            gradients.set_b(two_measurements, dt, invalid)
    for invalid in ([1e9], [1e9, 2e9, 3e9], [[1e9], [2e9]]):
        with pytest.raises(ValueError):
            gradients.set_b(two_measurements, dt, invalid)
    with pytest.raises(ValueError, match="zero waveform"):
        gradients.set_b(np.zeros_like(gradient), dt, 1e9)


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


def test_rotate_gradient_requires_one_matrix_per_measurement():
    gradient, _ = example_gradient()
    two_measurements = np.repeat(gradient, 2, axis=0)
    for Rs in (
        np.eye(3)[None, :, :],
        np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        np.eye(3),
    ):
        with pytest.raises(ValueError, match="number of measurements"):
            gradients.rotate_gradient(two_measurements, Rs)
    npt.assert_array_equal(
        gradients.rotate_gradient(
            two_measurements, np.repeat(np.eye(3)[None, :, :], 2, axis=0)
        ),
        two_measurements,
    )


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


def test_pgse_scalar_bvalue_and_antiparallel_direction():
    bvalue = 1e9
    gradient, dt = gradients.pgse(15e-3, 50e-3, 256, bvalue, np.array([-1.0, 0.0, 0.0]))
    npt.assert_equal(gradient.shape, (1, 256, 3))
    npt.assert_allclose(gradients.calc_b(gradient, dt), [bvalue], rtol=1e-12)
    npt.assert_allclose(gradient[0, 1, 0], -np.linalg.norm(gradient[0, 1]), atol=1e-14)


@pytest.mark.parametrize("n_t", [256, 257])
def test_pgse_touching_pulses_without_reference_array(monkeypatch, n_t):
    original_zeros = np.zeros

    def bounded_zeros(shape, *args, **kwargs):
        if np.prod(shape) > 100_000:
            raise AssertionError("pgse allocated its former million-sample array")
        return original_zeros(shape, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(gradients.np, "zeros", bounded_zeros)
        gradient, dt = gradients.pgse(15e-3, 15e-3, n_t, 1e9, [1, 0, 0])

    npt.assert_equal(gradient.shape, (1, n_t, 3))
    npt.assert_allclose(gradient[0, :, 0], -gradient[0, ::-1, 0], atol=1e-10)
    npt.assert_allclose(gradients.calc_b(gradient, dt), [1e9], rtol=1e-12)


def test_pgse_b0_zero_direction_and_mixed_measurements():
    bvals = [0, 1e9]
    bvecs = [[0, 0, 0], [0, 1, 0]]
    gradient, dt = gradients.pgse(15e-3, 50e-3, 256, bvals, bvecs)
    npt.assert_array_equal(gradient[0], 0)
    npt.assert_allclose(gradients.calc_b(gradient, dt), bvals, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(
        gradient[1, 1] / np.linalg.norm(gradient[1, 1]), bvecs[1], atol=1e-14
    )


def test_pgse_invalid_inputs_and_underresolved_pulses():
    for bvals in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            gradients.pgse(15e-3, 50e-3, 256, bvals, [1, 0, 0])
    with pytest.raises(ValueError, match="nonzero b-vectors"):
        gradients.pgse(15e-3, 50e-3, 256, 1e9, [0, 0, 0])
    with pytest.raises(ValueError, match="one three-dimensional b-vector"):
        gradients.pgse(15e-3, 50e-3, 256, [0, 1e9], [[0, 0, 0]])
    with pytest.raises(ValueError, match="under-resolved"):
        gradients.pgse(15e-3, 50e-3, 8, 1e9, [1, 0, 0])


def test_load_camino_scheme_file():
    gradient = np.zeros((1, 100, 3))
    gradient[0, 1:30, 0] = 1
    gradient[0, 70:99, 0] = -1
    T = 80e-3
    n_t = int(1e3)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    bs = np.linspace(0, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient = gradients.set_b(gradient, dt, bs)
    gradient_camino, dt_camino = gradients.load_camino_scheme_file(
        os.path.join(os.path.dirname(gradients.__file__), "tests", "camino.scheme")
    )
    npt.assert_almost_equal(gradient_camino, gradient)
    npt.assert_almost_equal(dt_camino, dt)
    return
