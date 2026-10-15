"""Fixed-seed equivalence of fused and one-step CUDA diffusion kernels."""

import numpy as np
import pytest

from .. import simulations, substrates
from .._cuda import (
    _RNG_STATE_DTYPE,
    IS_MLIR,
    _cuda_fused_free_large_phase,
    _cuda_fused_free_small_phase,
    _cuda_fused_sphere_small_phase,
    _cuda_step_free,
    _cuda_step_sphere,
    cuda,
    empty_xoroshiro128p_states,
)


def _splitmix_states(n, seed):
    """Independent CPU reference for the diffusion stream's initial state."""
    indices = np.arange(n, dtype=np.uint64)
    gamma = np.uint64(0x9E3779B97F4A7C15)

    def mixed(values):
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))

    with np.errstate(over="ignore"):
        states = np.empty(n, dtype=_RNG_STATE_DTYPE)
        values = np.uint64(seed) + (2 * indices + np.uint64(1)) * gamma
        states["s0"] = mixed(values.copy())
        states["s1"] = mixed(values + gamma)
    return states


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("channels", [3, 20])
@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_fused_free_matches_one_step(channels, chunk_size):
    rng = np.random.default_rng(41)
    n_walkers = 19
    n_samples = 9
    seed = 614
    step_l = 2.5e-6
    dt = 2e-4
    gradients = rng.normal(size=(channels, n_samples, 3)) * 0.1
    positions = rng.normal(size=(n_walkers, 3)) * 2e-6
    block_size = 32
    grid_size = 1
    stream = cuda.stream()
    d_gradient = [
        cuda.to_device(np.ascontiguousarray(gradients[:, :, axis]), stream=stream)
        for axis in range(3)
    ]
    d_reference_positions = cuda.to_device(positions, stream=stream)
    d_fused_positions = cuda.to_device(positions, stream=stream)
    d_reference_phases = cuda.to_device(np.zeros((channels, n_walkers)), stream=stream)
    d_fused_phases = cuda.to_device(np.zeros((channels, n_walkers)), stream=stream)
    reference_states = cuda.to_device(
        _splitmix_states(grid_size * block_size, seed), stream=stream
    )
    fused_states = empty_xoroshiro128p_states(grid_size * block_size, stream)
    for t in range(n_samples - 1):
        _cuda_step_free[grid_size, block_size, stream](
            d_reference_positions,
            *d_gradient,
            d_reference_phases,
            reference_states,
            t,
            step_l,
            dt,
        )
    fused_kernel = (
        _cuda_fused_free_small_phase if channels <= 16 else _cuda_fused_free_large_phase
    )
    for start in range(0, n_samples - 1, chunk_size):
        fused_kernel[grid_size, block_size, stream](
            d_fused_positions,
            *d_gradient,
            d_fused_phases,
            fused_states,
            start,
            min(start + chunk_size, n_samples - 1),
            seed,
            start == 0,
            step_l,
            dt,
        )
    stream.synchronize()
    np.testing.assert_allclose(
        d_fused_positions.copy_to_host(),
        d_reference_positions.copy_to_host(),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        d_fused_phases.copy_to_host(),
        d_reference_phases.copy_to_host(),
        rtol=0,
        atol=1e-11,
    )
    reference = reference_states.copy_to_host()[:n_walkers]
    fused = fused_states.copy_to_host()[:n_walkers]
    np.testing.assert_array_equal(fused, reference)


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("chunk_size", [1, 4, 8])
def test_fused_sphere_matches_one_step(chunk_size):
    rng = np.random.default_rng(87)
    n_walkers = 19
    n_samples = 9
    channels = 3
    seed = 712
    radius = 4e-6
    step_l = 2.5e-6
    dt = 2e-4
    epsilon = 1e-13
    max_iter = 100
    gradients = rng.normal(size=(channels, n_samples, 3)) * 0.1
    positions = rng.normal(size=(n_walkers, 3))
    positions *= (radius * 0.7 / np.linalg.norm(positions, axis=1))[:, None]
    block_size = 32
    grid_size = 1
    stream = cuda.stream()
    d_gradient = [
        cuda.to_device(np.ascontiguousarray(gradients[:, :, axis]), stream=stream)
        for axis in range(3)
    ]
    d_reference_positions = cuda.to_device(positions, stream=stream)
    d_fused_positions = cuda.to_device(positions, stream=stream)
    d_reference_phases = cuda.to_device(np.zeros((channels, n_walkers)), stream=stream)
    d_fused_phases = cuda.to_device(np.zeros((channels, n_walkers)), stream=stream)
    reference_states = cuda.to_device(
        _splitmix_states(grid_size * block_size, seed), stream=stream
    )
    fused_states = empty_xoroshiro128p_states(grid_size * block_size, stream)
    reference_flags = cuda.to_device(np.zeros(n_walkers, dtype=np.bool_), stream=stream)
    fused_flags = cuda.to_device(np.zeros(n_walkers, dtype=np.bool_), stream=stream)
    for t in range(n_samples - 1):
        _cuda_step_sphere[grid_size, block_size, stream](
            d_reference_positions,
            *d_gradient,
            d_reference_phases,
            reference_states,
            t,
            step_l,
            dt,
            radius,
            reference_flags,
            max_iter,
            epsilon,
        )
    for start in range(0, n_samples - 1, chunk_size):
        _cuda_fused_sphere_small_phase[grid_size, block_size, stream](
            d_fused_positions,
            *d_gradient,
            d_fused_phases,
            fused_states,
            start,
            min(start + chunk_size, n_samples - 1),
            seed,
            start == 0,
            step_l,
            dt,
            radius,
            fused_flags,
            max_iter,
            epsilon,
        )
    stream.synchronize()
    np.testing.assert_allclose(
        d_fused_positions.copy_to_host(),
        d_reference_positions.copy_to_host(),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        d_fused_phases.copy_to_host(),
        d_reference_phases.copy_to_host(),
        rtol=0,
        atol=1e-11,
    )
    np.testing.assert_array_equal(
        fused_states.copy_to_host()[:n_walkers],
        reference_states.copy_to_host()[:n_walkers],
    )
    np.testing.assert_array_equal(
        fused_flags.copy_to_host(), reference_flags.copy_to_host()
    )


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("kind", ["free", "sphere", "cylinder", "ellipsoid"])
def test_public_trajectory_matches_fused_seeded_output(monkeypatch, tmp_path, kind):
    rng = np.random.default_rng(107)
    gradient = rng.normal(size=(12, 1, 3)) * rng.normal(size=(1, 20, 1)) * 0.02
    if kind == "free":
        substrate = substrates.free()
    elif kind == "sphere":
        substrate = substrates.sphere(5e-6)
    elif kind == "cylinder":
        substrate = substrates.cylinder(5e-6, np.array([0.4, 0.7, 0.6]))
    else:
        substrate = substrates.ellipsoid(np.array([3e-6, 5e-6, 8e-6]))
    options = {
        "n_walkers": 19,
        "diffusivity": 2e-9,
        "gradient": gradient,
        "dt": 1e-4,
        "substrate": substrate,
        "seed": 42,
        "quiet": True,
        "final_pos": True,
        "all_signals": True,
    }
    monkeypatch.setattr(simulations, "_step_chunk_size", lambda *args: 8)
    fused_signal, fused_positions = simulations.simulation(**options)
    trajectory_path = tmp_path / f"{kind}.txt"
    trajectory_signal, trajectory_positions = simulations.simulation(
        **options, traj=str(trajectory_path)
    )
    trajectory = np.loadtxt(trajectory_path).reshape(20, 19, 3)
    np.testing.assert_allclose(
        fused_positions, trajectory_positions, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(fused_signal, trajectory_signal, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(trajectory[-1], trajectory_positions)
