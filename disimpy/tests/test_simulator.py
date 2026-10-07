"""Small simulations for CPU execution through the CUDA simulator."""

import os

import numpy as np
import pytest

from .. import simulations, substrates


pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_ENABLE_CUDASIM") != "1",
    reason="requires NUMBA_ENABLE_CUDASIM=1 before import",
)


def test_free_diffusion():
    gradient = np.zeros((2, 4, 3), dtype=np.float64)
    gradient[1, 1:3, 0] = 0.5
    options = dict(
        n_walkers=4,
        diffusivity=2e-9,
        gradient=gradient,
        dt=1e-3,
        substrate=substrates.free(),
        seed=123,
        final_pos=True,
        all_signals=True,
        quiet=True,
        cuda_bs=4,
    )

    signals, positions = simulations.simulation(**options)
    repeated_signals, repeated_positions = simulations.simulation(**options)

    assert signals.shape == (2, 4)
    assert positions.shape == (4, 3)
    np.testing.assert_array_equal(signals[0], np.ones(4))
    assert np.all(np.isfinite(signals))
    assert np.all(np.abs(signals) <= 1)
    assert np.any(signals[1] < 1)
    np.testing.assert_array_equal(signals, repeated_signals)
    np.testing.assert_array_equal(positions, repeated_positions)


@pytest.mark.parametrize(
    ("substrate", "radii"),
    [
        (substrates.sphere(5e-6), np.array([5e-6, 5e-6, 5e-6])),
        (
            substrates.cylinder(5e-6, np.array([1.0, 0.0, 0.0])),
            np.array([np.inf, 5e-6, 5e-6]),
        ),
        (
            substrates.ellipsoid(np.array([4e-6, 5e-6, 6e-6])),
            np.array([4e-6, 5e-6, 6e-6]),
        ),
    ],
    ids=["sphere", "cylinder", "ellipsoid"],
)
def test_restricted_diffusion(substrate, radii, tmp_path):
    trajectory_path = tmp_path / "trajectory.txt"
    options = dict(
        n_walkers=4,
        diffusivity=2e-9,
        gradient=np.zeros((1, 4, 3), dtype=np.float64),
        dt=1e-3,
        substrate=substrate,
        seed=123,
        traj=str(trajectory_path),
        final_pos=True,
        quiet=True,
        cuda_bs=4,
    )

    signals, positions = simulations.simulation(**options)
    repeated_signals, repeated_positions = simulations.simulation(**options)
    trajectory = np.loadtxt(trajectory_path).reshape(5, 4, 3)

    assert signals.shape == (1,)
    assert positions.shape == (4, 3)
    np.testing.assert_array_equal(signals, np.array([4.0]))
    np.testing.assert_array_equal(positions, trajectory[-1])
    np.testing.assert_array_equal(signals, repeated_signals)
    np.testing.assert_array_equal(positions, repeated_positions)
    assert np.all(np.linalg.norm(trajectory / radii, axis=2) <= 1)
