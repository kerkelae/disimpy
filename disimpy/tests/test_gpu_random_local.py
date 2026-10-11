"""Chunk-local RNG must reproduce the existing per-draw device sequence."""

import numpy as np
import pytest

from .._cuda import IS_MLIR, cuda
from .._cuda import (
    create_xoroshiro128p_states,
    derive_seed,
    xoroshiro128p_uniform_float64,
    xoroshiro128p_unit_sphere_float64,
)
from .._cuda import (
    create_local_xoroshiro128p_states,
    empty_xoroshiro128p_states,
    load_local_rng,
    store_local_rng,
    xoroshiro128p_uniform_local_float64,
    xoroshiro128p_unit_sphere_local_float64,
)


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires CUDA GPU")
def test_structured_splitmix_initializer_matches_host_formula():
    mask = (1 << 64) - 1
    gamma = 0x9E3779B97F4A7C15

    def splitmix(value):
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    seed = derive_seed(37, 2)
    expected = np.array(
        [
            (
                splitmix((seed + (2 * index + 1) * gamma) & mask),
                splitmix((seed + (2 * index + 2) * gamma) & mask),
            )
            for index in range(7)
        ],
        dtype=np.uint64,
    )
    stream = cuda.stream()
    states = create_local_xoroshiro128p_states(7, seed=seed, stream=stream)
    initialized = states.copy_to_host(stream=stream)
    stream.synchronize()
    words = np.column_stack((initialized["s0"], initialized["s1"]))
    np.testing.assert_array_equal(words, expected)


@cuda.jit
def _draw_global_and_local(
    global_states, local_states, global_vectors, local_vectors, global_u, local_u
):
    thread = cuda.grid(1)
    if thread >= global_vectors.shape[0]:
        return
    state = cuda.local.array(2, np.uint64)
    load_local_rng(local_states, thread, state)
    for step in range(global_vectors.shape[1]):
        xoroshiro128p_unit_sphere_float64(
            global_states, thread, global_vectors[thread, step]
        )
        xoroshiro128p_unit_sphere_local_float64(state, local_vectors[thread, step])
        global_u[thread, step] = xoroshiro128p_uniform_float64(global_states, thread)
        local_u[thread, step] = xoroshiro128p_uniform_local_float64(state)
    store_local_rng(local_states, thread, state)


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires CUDA GPU")
def test_local_rng_matches_global_draws_and_final_state():
    n_walkers, n_steps = 7, 5
    stream = cuda.stream()
    global_states = create_xoroshiro128p_states(n_walkers, seed=71, stream=stream)
    local_states = create_xoroshiro128p_states(n_walkers, seed=71, stream=stream)
    empty_states = empty_xoroshiro128p_states(n_walkers, stream)
    assert empty_states.shape == global_states.shape
    assert empty_states.dtype == global_states.dtype
    shape = (n_walkers, n_steps)
    global_vectors = cuda.device_array((*shape, 3), dtype=np.float64, stream=stream)
    local_vectors = cuda.device_array((*shape, 3), dtype=np.float64, stream=stream)
    global_u = cuda.device_array(shape, dtype=np.float64, stream=stream)
    local_u = cuda.device_array(shape, dtype=np.float64, stream=stream)
    _draw_global_and_local[1, 8, stream](
        global_states,
        local_states,
        global_vectors,
        local_vectors,
        global_u,
        local_u,
    )
    np.testing.assert_allclose(
        global_vectors.copy_to_host(stream=stream),
        local_vectors.copy_to_host(stream=stream),
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_array_equal(
        global_u.copy_to_host(stream=stream), local_u.copy_to_host(stream=stream)
    )
    np.testing.assert_array_equal(
        global_states.copy_to_host(stream=stream),
        local_states.copy_to_host(stream=stream),
    )
    stream.synchronize()
