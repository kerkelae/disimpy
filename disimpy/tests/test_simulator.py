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
    np.testing.assert_allclose(
        signals[1],
        [0.91299501, 0.94767009, 0.52637601, 0.99036159],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(signals, repeated_signals)
    np.testing.assert_array_equal(positions, repeated_positions)


def test_real_signal_matches_phase_definition():
    gradient = np.zeros((1, 2, 3))
    gradient[0, :, 0] = 0.1
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
    real_signal, positions = simulations.simulation(**options)
    expected = np.cos(267.513e6 * options["dt"] * 0.1 * positions[:, 0])
    np.testing.assert_allclose(real_signal[0], expected, rtol=0, atol=1e-14)


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
    trajectory = np.loadtxt(trajectory_path).reshape(4, 4, 3)

    assert signals.shape == (1,)
    assert positions.shape == (4, 3)
    np.testing.assert_array_equal(signals, np.array([4.0]))
    np.testing.assert_array_equal(positions, trajectory[-1])
    np.testing.assert_array_equal(signals, repeated_signals)
    np.testing.assert_array_equal(positions, repeated_positions)
    assert np.all(np.linalg.norm(trajectory / radii, axis=2) <= 1)


@pytest.mark.parametrize(
    "substrate",
    [
        substrates.sphere(1.0),
        substrates.cylinder(1.0, np.array([1.0, 0.0, 0.0])),
        substrates.ellipsoid(np.ones(3)),
    ],
    ids=["sphere", "cylinder", "ellipsoid"],
)
def test_max_iter_one_does_not_flag_collision_free_step(substrate):
    signals = simulations.simulation(
        n_walkers=8,
        diffusivity=1e-30,
        gradient=np.zeros((1, 2, 3)),
        dt=1e-3,
        substrate=substrate,
        seed=123,
        max_iter=1,
        quiet=True,
    )
    np.testing.assert_array_equal(signals, [8.0])


@pytest.mark.parametrize(
    "substrate",
    [
        substrates.sphere(1e-9),
        substrates.cylinder(1e-9, np.array([1.0, 0.0, 0.0])),
        substrates.ellipsoid(np.full(3, 1e-9)),
    ],
    ids=["sphere", "cylinder", "ellipsoid"],
)
def test_unresolved_collision_raises(substrate):
    with pytest.raises(RuntimeError, match="Collision resolution exceeded max_iter=1"):
        simulations.simulation(
            n_walkers=32,
            diffusivity=2e-9,
            gradient=np.zeros((1, 2, 3)),
            dt=1e-3,
            substrate=substrate,
            seed=123,
            max_iter=1,
            quiet=True,
        )
