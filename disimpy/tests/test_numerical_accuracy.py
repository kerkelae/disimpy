"""Finite-step physics checks through the CUDA simulator."""

import math
import os

import numpy as np
import pytest

from .. import gradients, simulations, substrates

pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_ENABLE_CUDASIM") != "1",
    reason="requires NUMBA_ENABLE_CUDASIM=1 before import",
)


def test_free_fixed_step_signal_and_msd_match_exact_discrete_results():
    n_walkers = 1024
    n_steps = 3
    diffusivity = 2e-9
    dt = 1e-3
    step_length = np.sqrt(6 * diffusivity * dt)

    # A constant gradient makes the three independent step-direction factors
    # sinc(3a), sinc(2a), and sinc(a) for the post-step phase convention.
    a = 0.2
    gradient = np.zeros((1, n_steps + 1, 3))
    gradient[0, :, 0] = a / (gradients.GAMMA * dt * step_length)
    exact_signal = np.prod(np.sinc(a * np.arange(n_steps, 0, -1) / np.pi))

    signal, positions = simulations.simulation(
        n_walkers=n_walkers,
        diffusivity=diffusivity,
        gradient=gradient,
        dt=dt,
        substrate=substrates.free(),
        seed=123,
        final_pos=True,
        quiet=True,
        cuda_bs=128,
    )

    np.testing.assert_allclose(signal[0] / n_walkers, exact_signal, atol=0.025)
    exact_msd = n_steps * step_length**2
    measured_msd = np.mean(np.sum(positions**2, axis=1))
    np.testing.assert_allclose(measured_msd, exact_msd, rtol=0.08)


def _membrane_substrate(initial_positions, **permeability):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10e-6, 0.0],
            [0.0, 10e-6, 10e-6],
            [0.0, 0.0, 10e-6],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return substrates.mesh(
        vertices,
        faces,
        periodic=False,
        padding=np.array([5e-6, 1e-6, 1e-6]),
        init_pos=initial_positions,
        n_sv=np.array([3, 2, 2]),
        quiet=True,
        **permeability,
    )


def test_narrow_reflecting_slab_contains_steps_larger_than_gap():
    gap = 0.5e-6
    n_walkers = 64
    initial_positions = np.empty((n_walkers, 3))
    initial_positions[:, 0] = np.linspace(1.02 * gap, 1.98 * gap, n_walkers)
    initial_positions[:, 1] = np.linspace(0.2e-6, 9.8e-6, n_walkers)
    initial_positions[:, 2] = np.linspace(9.8e-6, 0.2e-6, n_walkers)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10e-6, 0.0],
            [0.0, 10e-6, 10e-6],
            [0.0, 0.0, 10e-6],
        ]
    )
    substrate = substrates.mesh(
        vertices,
        np.array([[0, 1, 2], [0, 2, 3]]),
        periodic=False,
        padding=np.array([gap, 0.0, 0.0]),
        init_pos=initial_positions,
        n_sv=np.array([4, 2, 2]),
        quiet=True,
    )
    diffusivity = 2e-9
    dt = 1e-3
    assert math.sqrt(6 * diffusivity * dt) > 6 * gap

    _, final = simulations.simulation(
        n_walkers=n_walkers,
        diffusivity=diffusivity,
        gradient=np.zeros((1, 2, 3)),
        dt=dt,
        substrate=substrate,
        seed=321,
        final_pos=True,
        quiet=True,
        cuda_bs=32,
    )

    assert np.all(final[:, 0] >= gap)
    assert np.all(final[:, 0] <= 2 * gap)
