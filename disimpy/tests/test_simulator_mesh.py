"""Small mesh simulation through the CUDA simulator."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from .. import _cuda, _mesh, simulations, substrates

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

    path = np.loadtxt(trajectory).reshape(time_points, n_walkers, 3)
    np.testing.assert_array_equal(path[-1], positions)
    assert np.all(np.linalg.norm(path - center, axis=-1) <= radius + 1e-12)


def test_float32_explicit_positions_use_float64_and_traj_does_not_change_result(
    tmp_path, monkeypatch
):
    # Force two GPU-to-host trajectory flushes for four one-step samples.
    monkeypatch.setattr(_cuda, "_TRAJECTORY_BUFFER_BYTES", 2 * 4 * 3 * 8)
    vertices, faces = _mesh._aabb_to_mesh(np.full(3, 4e-6), np.full(3, 16e-6))
    initial = np.random.default_rng(42).uniform(7e-6, 13e-6, (4, 3)).astype(np.float32)
    gradient = np.zeros((2, 5, 3))
    gradient[1, :, 0] = 0.03
    options = dict(
        n_walkers=4,
        diffusivity=2e-9,
        gradient=gradient,
        dt=1e-4,
        seed=44,
        final_pos=True,
        all_signals=True,
        quiet=True,
        cuda_bs=4,
    )
    results = {}
    for dtype in (np.float32, np.float64):
        substrate = substrates.mesh(
            vertices,
            faces,
            periodic=False,
            padding=np.full(3, 4e-6),
            init_pos=initial.astype(dtype),
            n_sv=np.array([2, 2, 2]),
            quiet=True,
        )
        for record in (False, True):
            path = tmp_path / f"{np.dtype(dtype).name}_{record}.txt"
            signal, final = simulations.simulation(
                **options, substrate=substrate, traj=str(path) if record else None
            )
            assert final.dtype == np.float64
            if record:
                trajectory = np.loadtxt(path).reshape(5, 4, 3)
                np.testing.assert_array_equal(trajectory[0], initial.astype(np.float64))
                np.testing.assert_array_equal(trajectory[-1], final)
            results[(dtype, record)] = signal, final

    reference = results[(np.float64, False)]
    for signal, final in results.values():
        np.testing.assert_array_equal(signal, reference[0])
        np.testing.assert_array_equal(final, reference[1])


@pytest.mark.parametrize("perm_prob", [0.0, 1.0])
def test_nonperiodic_voxel_walls_are_impermeable(perm_prob):
    # The small interior triangle is permeable at perm_prob=1, but the six
    # artificial voxel walls must reflect independently of that probability.
    vertices = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 1e-6]])
    initial_positions = np.tile([10.9e-6, 5.5e-6, 5.5e-6], (64, 1))
    substrate = substrates.mesh(
        vertices,
        np.array([[0, 1, 2]]),
        periodic=False,
        padding=np.full(3, 5e-6),
        init_pos=initial_positions,
        n_sv=np.array([2, 2, 2]),
        perm_prob=perm_prob,
        quiet=True,
    )
    assert substrate.n_membrane_faces == 1

    _, final_positions = simulations.simulation(
        n_walkers=len(initial_positions),
        diffusivity=2e-9,
        gradient=np.zeros((1, 3, 3)),
        dt=1e-3,
        substrate=substrate,
        seed=123,
        final_pos=True,
        quiet=True,
    )

    assert np.all(final_positions >= -1e-12)
    assert np.all(final_positions <= substrate.voxel_size + 1e-12)


@pytest.mark.parametrize("fused", [False, True])
def test_exact_voxel_wall_starts_reflect_outward_steps(fused, monkeypatch):
    # A zero-distance triangle hit is deliberately excluded after a collision,
    # but explicit initial positions may lie exactly on either voxel wall.
    vertices = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 1e-6]])
    padding = np.full(3, 5e-6)
    voxel_size = np.max(vertices + padding, axis=0) + padding
    initial_positions = np.tile([0.0, 5.5e-6, 5.5e-6], (64, 1))
    initial_positions[32:, 0] = voxel_size[0]
    substrate = substrates.mesh(
        vertices,
        np.array([[0, 1, 2]]),
        periodic=False,
        padding=padding,
        init_pos=initial_positions,
        n_sv=np.array([2, 2, 2]),
        quiet=True,
    )
    if fused:
        # Force the fused kernel for a single step so both execution paths
        # check the same boundary condition with a small simulator workload.
        monkeypatch.setattr(simulations, "_step_chunk_size", lambda *args: 64)

    _, final_positions = simulations.simulation(
        n_walkers=len(initial_positions),
        diffusivity=2e-9,
        gradient=np.zeros((1, 2, 3)),
        dt=1e-3,
        substrate=substrate,
        seed=123,
        final_pos=True,
        quiet=True,
        cuda_bs=64,
    )

    assert np.all(final_positions >= 0)
    assert np.all(final_positions <= substrate.voxel_size)


def test_membrane_permeability_still_controls_crossing():
    # A two-triangle plane spans the possible step endpoints. Its central
    # location avoids interactions with the artificial box walls.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 10e-6, 0.0], [0.0, 10e-6, 10e-6], [0.0, 0.0, 10e-6]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    initial_positions = np.tile([4.9e-6, 5.9e-6, 6.2e-6], (64, 1))
    final_positions = []
    for perm_prob in (0.0, 1.0):
        substrate = substrates.mesh(
            vertices,
            faces,
            periodic=False,
            padding=np.array([5e-6, 1e-6, 1e-6]),
            init_pos=initial_positions,
            n_sv=np.array([3, 2, 2]),
            perm_prob=perm_prob,
            quiet=True,
        )
        _, positions = simulations.simulation(
            n_walkers=len(initial_positions),
            diffusivity=2e-9,
            gradient=np.zeros((1, 2, 3)),
            dt=1e-3,
            substrate=substrate,
            seed=123,
            final_pos=True,
            quiet=True,
        )
        final_positions.append(positions)

    assert np.all(final_positions[0][:, 0] < 5e-6)
    assert np.any(final_positions[1][:, 0] > 5e-6)


def test_mesh_unresolved_collision_raises():
    vertices = np.array([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, 1e-9, 1e-9]])
    initial_positions = np.tile([1.5e-9, 1.5e-9, 1.5e-9], (32, 1))
    substrate = substrates.mesh(
        vertices,
        np.array([[0, 1, 2]]),
        periodic=False,
        padding=np.full(3, 1e-9),
        init_pos=initial_positions,
        n_sv=np.array([2, 2, 2]),
        quiet=True,
    )
    with pytest.raises(RuntimeError, match="Collision resolution exceeded max_iter=1"):
        simulations.simulation(
            n_walkers=len(initial_positions),
            diffusivity=5e-15,
            gradient=np.zeros((1, 2, 3)),
            dt=1e-3,
            substrate=substrate,
            seed=123,
            max_iter=1,
            quiet=True,
        )
