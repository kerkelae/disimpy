"""Small mesh simulation through the CUDA simulator."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from .. import simulations, substrates


pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_ENABLE_CUDASIM") != "1",
    reason="requires NUMBA_ENABLE_CUDASIM=1 before import",
)


def test_mesh_confinement(tmp_path):
    with Path(__file__).with_name("sphere_mesh.pkl").open("rb") as handle:
        mesh = pickle.load(handle)

    vertices = mesh["vertices"]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    radius = (vertices.max(axis=0) - vertices.min(axis=0))[0] / 2
    n_walkers = 4
    time_points = 4
    initial_positions = np.tile(center, (n_walkers, 1))
    substrate = substrates.mesh(
        vertices,
        mesh["faces"],
        periodic=False,
        init_pos=initial_positions,
        n_sv=np.array([1, 1, 1]),
        quiet=True,
    )
    gradient = np.zeros((2, time_points, 3))
    gradient[1, 1:3, 0] = 0.5
    trajectory = tmp_path / "mesh_trajectory.txt"

    signals, positions = simulations.simulation(
        n_walkers=n_walkers,
        diffusivity=2e-9,
        gradient=gradient,
        dt=1e-3,
        substrate=substrate,
        seed=123,
        traj=str(trajectory),
        final_pos=True,
        all_signals=True,
        quiet=True,
        cuda_bs=4,
    )

    assert signals.shape == (2, n_walkers)
    assert positions.shape == (n_walkers, 3)
    np.testing.assert_array_equal(signals[0], np.ones(n_walkers))
    assert np.all(np.isfinite(signals))
    assert np.all(np.abs(signals) <= 1)

    path = np.loadtxt(trajectory).reshape(time_points + 1, n_walkers, 3)
    np.testing.assert_array_equal(path[-1], positions)
    assert np.all(np.linalg.norm(path - center, axis=-1) <= radius + 1e-12)
