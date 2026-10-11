"""Seeded RNG regression checks across simulator and MLIR backends."""

import os

import numpy as np
import pytest

from .. import _cuda
from .._cuda import IS_MLIR, cuda


@cuda.jit
def _draw_random(states, uniform, normal):
    i = cuda.grid(1)
    if i < uniform.size:
        uniform[i] = _cuda.xoroshiro128p_uniform_float64(states, i)
        normal[i] = _cuda.xoroshiro128p_normal_float64(states, i)


@cuda.jit
def _draw_unit_vectors(states, vectors):
    i = cuda.grid(1)
    if i < vectors.shape[0]:
        _cuda.xoroshiro128p_unit_sphere_float64(states, i, vectors[i])


def _draw_seeded(seed, stream):
    states = _cuda.create_xoroshiro128p_states(4, seed=seed, stream=stream)
    uniform = cuda.to_device(np.zeros(4, dtype=np.float64), stream=stream)
    normal = cuda.to_device(np.zeros(4, dtype=np.float64), stream=stream)
    _draw_random[1, 4, stream](states, uniform, normal)
    host_uniform = uniform.copy_to_host(stream=stream)
    host_normal = normal.copy_to_host(stream=stream)
    stream.synchronize()
    return host_uniform, host_normal


def test_derived_rng_domains_are_distinct_and_repeatable():
    seeds = [_cuda.derive_seed(123, domain) for domain in range(4)]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [_cuda.derive_seed(123, domain) for domain in range(4)]
    assert seeds != [_cuda.derive_seed(124, domain) for domain in range(4)]


@pytest.mark.skipif(not IS_MLIR, reason="tests MLIR host state initialization")
def test_mlir_state_seed_and_subsequence():
    states = _cuda._initialize_states(4, np.uint64(123), 0)
    expected = np.array(
        [
            (13032462758197477675, 13032462758197477675),
            (8675627150879738035, 7601652969287324781),
            (2915451743979457534, 18405070019864801170),
            (8160601399947100842, 1643275622517175902),
        ],
        dtype=states.dtype,
    )
    np.testing.assert_array_equal(states, expected)
    offset = _cuda._initialize_states(1, np.uint64(123), 1)
    np.testing.assert_array_equal(offset[0], states[1])
    repeated = _cuda._initialize_states(4, np.uint64(123), 0)
    np.testing.assert_array_equal(states, repeated)


@pytest.mark.skipif(not IS_MLIR, reason="tests MLIR vectorized state initialization")
def test_splitmix_state_initialization_is_repeatable_distinct_and_offsettable():
    states = _cuda._initialize_states_splitmix(8192, np.uint64(123), 0)
    repeated = _cuda._initialize_states_splitmix(8192, np.uint64(123), 0)
    changed_seed = _cuda._initialize_states_splitmix(8192, np.uint64(124), 0)
    offset = _cuda._initialize_states_splitmix(8191, np.uint64(123), 1)

    np.testing.assert_array_equal(states, repeated)
    np.testing.assert_array_equal(states[1:], offset)
    assert not np.array_equal(states, changed_seed)
    assert not np.any((states["s0"] == 0) & (states["s1"] == 0))
    assert len(np.unique(states["s0"])) == len(states)
    assert len(np.unique(states["s1"])) == len(states)


def _assert_isotropic_unit_vectors(
    vectors, mean_tolerance=0.025, moment_tolerance=0.02
):
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-15)
    assert np.max(np.abs(np.mean(vectors, axis=0))) < mean_tolerance
    second_moment = vectors.T @ vectors / len(vectors)
    np.testing.assert_allclose(
        np.diag(second_moment), np.full(3, 1 / 3), atol=moment_tolerance
    )
    off_diagonal = second_moment - np.diag(np.diag(second_moment))
    assert np.max(np.abs(off_diagonal)) < moment_tolerance
    # Coordinates from adjacent independently initialized streams should not
    # display a practically relevant linear correlation.
    for axis in range(3):
        assert abs(np.corrcoef(vectors[:-1, axis], vectors[1:, axis])[0, 1]) < 0.08


@pytest.mark.skipif(
    os.environ.get("NUMBA_ENABLE_CUDASIM") != "1",
    reason="statistical CPU test uses CUDA simulator",
)
def test_two_uniform_unit_vectors_are_isotropic_and_repeatable_in_cudasim():
    n = 4096
    first_states = _cuda.create_xoroshiro128p_states(n, seed=123)
    second_states = _cuda.create_xoroshiro128p_states(n, seed=123)
    first = np.empty((n, 3))
    second = np.empty((n, 3))
    _draw_unit_vectors[32, 128](first_states, first)
    _draw_unit_vectors[32, 128](second_states, second)

    np.testing.assert_array_equal(first, second)
    _assert_isotropic_unit_vectors(first, mean_tolerance=0.04, moment_tolerance=0.03)


@pytest.mark.gpu
@pytest.mark.skipif(not IS_MLIR, reason="tests physical GPU with MLIR")
def test_two_uniform_unit_vectors_are_isotropic_and_repeatable_on_gpu():
    if not cuda.is_available():
        pytest.skip("no physical CUDA device")
    n = 262144
    host_states = _cuda._initialize_states_splitmix(n, np.uint64(123), 0)
    first = cuda.device_array((n, 3), dtype=np.float64)
    second = cuda.device_array((n, 3), dtype=np.float64)
    _draw_unit_vectors[2048, 128](cuda.to_device(host_states), first)
    _draw_unit_vectors[2048, 128](cuda.to_device(host_states), second)
    first = first.copy_to_host()
    second = second.copy_to_host()

    np.testing.assert_array_equal(first, second)
    _assert_isotropic_unit_vectors(first)


@pytest.mark.skipif(IS_MLIR, reason="tests Numba-CUDA simulator adapter")
def test_simulator_seeded_random():
    first = _draw_seeded(123, cuda.stream())
    second = _draw_seeded(123, cuda.stream())
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(
        first[0],
        [
            0.4129824435274134,
            0.882393123422012,
            0.15578780074422116,
            0.5314692383268244,
        ],
        rtol=0,
        atol=1e-15,
    )
    assert np.all(np.isfinite(first[1]))

    mesh_stream = _draw_seeded(_cuda.derive_seed(123, 1), cuda.stream())
    repeated_mesh_stream = _draw_seeded(_cuda.derive_seed(123, 1), cuda.stream())
    for a, b in zip(mesh_stream, repeated_mesh_stream):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], mesh_stream[0])


@pytest.mark.gpu
@pytest.mark.skipif(not IS_MLIR, reason="tests physical GPU with MLIR")
def test_mlir_seeded_random_on_gpu():
    if not cuda.is_available():
        pytest.skip("no physical CUDA device")
    first = _draw_seeded(123, cuda.stream())
    second = _draw_seeded(123, cuda.stream())
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    changed = _draw_seeded(124, cuda.stream())
    assert not np.array_equal(first[0], changed[0])
    assert np.all((first[0] >= 0) & (first[0] < 1))
    assert np.all(np.isfinite(first[1]))
