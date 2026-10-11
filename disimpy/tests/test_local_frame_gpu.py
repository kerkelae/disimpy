"""Orientation and lab-frame output checks for local analytic coordinates."""

import numpy as np
import pytest

from .. import simulations, substrates, utils
from .._cuda import IS_MLIR, cuda


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("kind", ["cylinder", "ellipsoid"])
@pytest.mark.parametrize("rank", [3, 12])
def test_rotating_substrate_and_gradient_preserves_signal_and_positions(kind, rank):
    rng = np.random.default_rng(97)
    samples = 6
    if rank == 3:
        gradient_local = (
            rng.normal(size=(12, 1, 3)) * rng.normal(size=(1, samples, 1)) * 0.03
        )
    else:
        gradient_local = rng.normal(size=(12, samples, 3)) * 0.03
    if kind == "cylinder":
        direction = np.array([0.5, -0.3, 0.8])
        Q = utils.vec2vec_rotmat(direction, np.array([1.0, 0.0, 0.0])).T
        base_substrate = substrates.cylinder(5e-6, np.array([1.0, 0.0, 0.0]))
        rotated_substrate = substrates.cylinder(5e-6, direction)
    else:
        angle = 0.71
        Q = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        base_substrate = substrates.ellipsoid(np.array([4e-6, 5e-6, 6e-6]))
        rotated_substrate = substrates.ellipsoid(np.array([4e-6, 5e-6, 6e-6]), Q)
    options = {
        "n_walkers": 24,
        "diffusivity": 2e-9,
        "dt": 1e-4,
        "seed": 321,
        "quiet": True,
        "all_signals": True,
        "final_pos": True,
    }
    reference_signal, reference_position = simulations.simulation(
        **options, gradient=gradient_local, substrate=base_substrate
    )
    rotated_signal, rotated_position = simulations.simulation(
        **options,
        gradient=np.einsum("mtj,ij->mti", gradient_local, Q),
        substrate=rotated_substrate,
    )
    np.testing.assert_allclose(rotated_signal, reference_signal, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        rotated_position, reference_position @ Q.T, rtol=0, atol=1e-12
    )


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("kind", ["cylinder", "ellipsoid"])
def test_trajectory_rows_are_lab_frame(kind, tmp_path):
    theta = 0.4
    Q = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    if kind == "cylinder":
        Q = utils.vec2vec_rotmat(Q[:, 0], np.array([1.0, 0.0, 0.0])).T
        base = substrates.cylinder(5e-6, np.array([1.0, 0.0, 0.0]))
        rotated = substrates.cylinder(5e-6, Q[:, 0])
    else:
        base = substrates.ellipsoid(np.array([4e-6, 5e-6, 6e-6]))
        rotated = substrates.ellipsoid(np.array([4e-6, 5e-6, 6e-6]), Q)
    options = {
        "n_walkers": 8,
        "diffusivity": 2e-9,
        "gradient": np.zeros((1, 4, 3)),
        "dt": 1e-4,
        "seed": 42,
        "quiet": True,
        "final_pos": True,
    }
    base_path = tmp_path / "base.txt"
    rotated_path = tmp_path / "rotated.txt"
    _, base_final = simulations.simulation(
        **options, substrate=base, traj=str(base_path)
    )
    _, rotated_final = simulations.simulation(
        **options, substrate=rotated, traj=str(rotated_path)
    )
    base_rows = np.loadtxt(base_path).reshape(4, 8, 3)
    rotated_rows = np.loadtxt(rotated_path).reshape(4, 8, 3)
    np.testing.assert_allclose(rotated_rows, base_rows @ Q.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(base_rows[-1], base_final, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rotated_rows[-1], rotated_final, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("option", "bad"),
    [("final_pos", 1), ("all_signals", 1), ("traj", False)],
)
def test_output_options_fail_before_gpu_probe(monkeypatch, option, bad):
    def forbidden_probe():
        raise AssertionError("CUDA probe must not run for invalid input")

    monkeypatch.setattr(simulations.cuda, "detect", forbidden_probe)
    options = {
        "n_walkers": 2,
        "diffusivity": 2e-9,
        "gradient": np.zeros((1, 2, 3)),
        "dt": 1e-4,
        "substrate": substrates.free(),
        option: bad,
    }
    with pytest.raises(ValueError, match=option):
        simulations.simulation(**options)


@pytest.mark.parametrize(
    "initial",
    [
        np.zeros((2, 2)),
        np.zeros((3, 3)),
        np.full((2, 3), np.nan),
        np.full((2, 3), 2.0),
        np.zeros((2, 3), dtype=np.int64),
        "unknown",
        1,
    ],
)
def test_mutated_mesh_initial_positions_fail_before_gpu_probe(monkeypatch, initial):
    def forbidden_probe():
        raise AssertionError("CUDA probe must not run for invalid input")

    monkeypatch.setattr(simulations.cuda, "detect", forbidden_probe)
    fake_mesh = object.__new__(substrates._Substrate)
    fake_mesh.type = "mesh"
    fake_mesh.voxel_size = np.ones(3)
    fake_mesh.init_pos = initial
    with pytest.raises(ValueError, match="init_pos"):
        simulations.simulation(2, 2e-9, np.zeros((1, 2, 3)), 1e-4, fake_mesh)


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("kind", ["free", "sphere", "cylinder", "ellipsoid"])
def test_simulation_does_not_modify_numpy_global_rng(kind):
    if kind == "free":
        substrate = substrates.free()
    elif kind == "sphere":
        substrate = substrates.sphere(5e-6)
    elif kind == "cylinder":
        substrate = substrates.cylinder(5e-6, np.array([1.0, 0.0, 0.0]))
    else:
        substrate = substrates.ellipsoid(np.array([5e-6] * 3))
    np.random.seed(417)
    expected = np.random.random(3)
    np.random.seed(417)
    simulations.simulation(
        8, 2e-9, np.zeros((1, 2, 3)), 1e-4, substrate, seed=1, quiet=True
    )
    np.testing.assert_array_equal(np.random.random(3), expected)
