"""CUDA backend, random streams, kernels, initialization, and reduction.

Set ``NUMBA_ENABLE_CUDASIM=1`` before importing disimpy to use Numba's CUDA
simulator instead of the physical-GPU MLIR backend.
"""

import math
import os

import numba
import numpy as np

from .gradients import GAMMA

IS_MLIR = os.environ.get("NUMBA_ENABLE_CUDASIM") != "1"
if IS_MLIR:
    from numba_cuda_mlir import cuda as cuda
else:
    from numba import cuda as cuda


# Global RNG and seed domains

"""Random states and device draws shared by the two CUDA backends.

The MLIR backend cannot compile the device functions in its compatibility
``cuda.random`` module (version 0.5.4). Its xoroshiro128+ state layout and
transition are retained here. Streams are initialized independently with
vectorized SplitMix64 expansion to avoid serial jump-ahead setup.

The xoroshiro128+ and SplitMix64 algorithms, constants, and state layout are
adapted from NVIDIA's BSD-2-Clause ``numba_cuda_mlir.numba_cuda.random``.
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def derive_seed(seed, domain):
    """Derive a deterministic 64-bit seed for an independent RNG domain."""
    mask = (1 << 64) - 1
    z = (int(seed) + (int(domain) + 1) * 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


if not IS_MLIR:
    from numba.cuda.random import (  # noqa: F401
        create_xoroshiro128p_states,
        xoroshiro128p_normal_float64,
        xoroshiro128p_uniform_float64,
    )
else:
    _STATE_DTYPE = np.dtype([("s0", np.uint64), ("s1", np.uint64)], align=True)

    def _splitmix64_array(values):
        """Apply the SplitMix64 output mixer to a uint64 NumPy array."""
        values = np.asarray(values, dtype=np.uint64)
        mixed = values ^ (values >> np.uint64(30))
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        return mixed ^ (mixed >> np.uint64(31))

    def _initialize_states_splitmix(n, seed, subsequence_start=0):
        """Initialize independent xoroshiro128+ streams without serial jumps.

        Each stream receives two consecutive outputs from SplitMix64, the
        initialization generator recommended for xoroshiro128+. Stream ``i``
        is keyed by ``subsequence_start + i`` so the work is vectorizable and
        deterministic. Unlike jump initialization, this does not promise
        disjoint 2**64-value subsequences; its statistical suitability is
        covered by the RNG validation benchmark and tests.
        """
        states = np.empty(n, dtype=_STATE_DTYPE)
        if n == 0:
            return states
        gamma = np.uint64(0x9E3779B97F4A7C15)
        stream_indices = np.arange(n, dtype=np.uint64)
        stream_indices += np.uint64(subsequence_start)
        splitmix_states = np.full(n, np.uint64(seed), dtype=np.uint64)
        splitmix_states += ((stream_indices << np.uint64(1)) + np.uint64(1)) * gamma
        states["s0"] = _splitmix64_array(splitmix_states)
        states["s1"] = _splitmix64_array(splitmix_states + gamma)
        # xoroshiro128+ has a single forbidden all-zero state. The chance is
        # 2**-128, but handle it deterministically rather than relying on odds.
        all_zero = (states["s0"] == 0) & (states["s1"] == 0)
        states["s1"][all_zero] = gamma
        return states

    @numba.njit
    def _rotate_left(x, count):
        return (x << np.uint32(count)) | (x >> np.uint32(64 - count))

    @numba.njit
    def _advance(states, index):
        s0 = states[index]["s0"]
        s1 = states[index]["s1"]
        result = s0 + s1
        s1 ^= s0
        states[index]["s0"] = _rotate_left(s0, 55) ^ s1 ^ (s1 << np.uint32(14))
        states[index]["s1"] = _rotate_left(s1, 36)
        return result

    @numba.njit
    def _jump(states, index):
        jump_constants = (
            np.uint64(0xBEAC0467EBA5FACB),
            np.uint64(0xD86B048B86AA9922),
        )
        s0 = np.uint64(0)
        s1 = np.uint64(0)
        for word in jump_constants:
            for bit in range(64):
                if word & (np.uint64(1) << np.uint32(bit)):
                    s0 ^= states[index]["s0"]
                    s1 ^= states[index]["s1"]
                _advance(states, index)
        states[index]["s0"] = s0
        states[index]["s1"] = s1

    @numba.njit
    def _initialize_states(n, seed, subsequence_start):
        states = np.empty(n, dtype=_STATE_DTYPE)
        if n == 0:
            return states

        z = seed + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint32(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint32(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint32(31)
        states[0]["s0"] = z
        states[0]["s1"] = z

        for _ in range(subsequence_start):
            _jump(states, 0)
        for i in range(1, n):
            states[i] = states[i - 1]
            _jump(states, i)
        return states

    def create_xoroshiro128p_states(n, seed, subsequence_start=0, stream=0):
        """Create independently seeded xoroshiro128+ states.

        SplitMix64 expansion is deterministic and fast, but does not
        mathematically guarantee the disjoint 2**64-value subsequences of the
        slower serial ``_initialize_states`` reference. Seeded MLIR sequences
        therefore differ from releases that used serial jump initialization.
        """
        states = _initialize_states_splitmix(n, np.uint64(seed), subsequence_start)
        return cuda.to_device(states, stream=stream)

    @cuda.jit(device=True)
    def _next_uint64(states, index):
        s0 = states[index]["s0"]
        s1 = states[index]["s1"]
        result = s0 + s1
        s1 ^= s0
        states[index]["s0"] = (
            ((s0 << np.uint32(55)) | (s0 >> np.uint32(9))) ^ s1 ^ (s1 << np.uint32(14))
        )
        states[index]["s1"] = (s1 << np.uint32(36)) | (s1 >> np.uint32(28))
        return result

    @cuda.jit(device=True)
    def xoroshiro128p_uniform_float64(states, index):
        """Draw a float64 in [0, 1) and advance the selected state."""
        value = _next_uint64(states, index)
        return np.float64(value >> np.uint32(11)) * (1.0 / 9007199254740992.0)

    @cuda.jit(device=True)
    def xoroshiro128p_normal_float64(states, index):
        """Draw a standard normal using two float32 uniforms, as Numba does."""
        u1 = np.float64(np.float32(xoroshiro128p_uniform_float64(states, index)))
        u2 = np.float64(np.float32(xoroshiro128p_uniform_float64(states, index)))
        return math.sqrt(-2.0 * math.log(u1)) * math.cos((2.0 * math.pi) * u2)


@cuda.jit(device=True)
def xoroshiro128p_unit_sphere_float64(states, index, vector):
    """Draw an exactly isotropic three-dimensional unit vector.

    Uniform ``z`` and azimuth give the surface-area measure on a sphere while
    consuming two uniforms. This avoids the six uniforms, three Box-Muller
    transforms, and normalization used by three independent normal draws.
    """
    z = 2.0 * xoroshiro128p_uniform_float64(states, index) - 1.0
    phi = 2.0 * math.pi * xoroshiro128p_uniform_float64(states, index)
    radial_squared = 1.0 - z * z
    if radial_squared < 0.0:
        radial_squared = 0.0
    radial = math.sqrt(radial_squared)
    vector[0] = radial * math.cos(phi)
    vector[1] = radial * math.sin(phi)
    vector[2] = z


# Local RNG state for fused kernels

"""Chunk-local xoroshiro128+ draws for fused diffusion kernels.

The transition, conversion to uniform float64, and spherical draw match
``_gpu_random``. Keeping two uint64 state words in per-thread local storage
lets fused kernels write the state to device memory once per chunk.

The xoroshiro128+ algorithm and constants are adapted from NVIDIA's BSD-2
Clause ``numba_cuda_mlir.numba_cuda.random``. Copyright (c) 2025 NVIDIA
CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


if IS_MLIR:
    _RNG_STATE_DTYPE = _STATE_DTYPE
else:
    from numba.cuda.random import xoroshiro128p_dtype as _RNG_STATE_DTYPE


def empty_xoroshiro128p_states(n, stream):
    """Allocate diffusion RNG state storage without a host initialization upload."""
    return cuda.device_array(n, dtype=_RNG_STATE_DTYPE, stream=stream)


@cuda.jit(device=True)
def _splitmix64_local(value):
    mixed = value ^ (value >> np.uint32(30))
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint32(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    return mixed ^ (mixed >> np.uint32(31))


@cuda.jit(device=True)
def initialize_local_rng(seed, index, local_state):
    """Match the vectorized SplitMix64 stream keyed by walker index."""
    gamma = np.uint64(0x9E3779B97F4A7C15)
    splitmix_state = (
        np.uint64(seed) + (np.uint64(index) * np.uint64(2) + np.uint64(1)) * gamma
    )
    local_state[0] = _splitmix64_local(splitmix_state)
    local_state[1] = _splitmix64_local(splitmix_state + gamma)
    if local_state[0] == 0 and local_state[1] == 0:
        local_state[1] = gamma


@cuda.jit
def _initialize_xoroshiro128p_states(states, seed):
    """Initialize structured device states with the fused-kernel stream map."""
    index = cuda.grid(1)
    if index >= states.shape[0]:
        return
    local_state = cuda.local.array(2, np.uint64)
    initialize_local_rng(seed, index, local_state)
    states[index]["s0"] = local_state[0]
    states[index]["s1"] = local_state[1]


def create_local_xoroshiro128p_states(n, seed, stream):
    """Create structured states matching first-chunk fused initialization."""
    states = empty_xoroshiro128p_states(n, stream)
    if n:
        threads_per_block = 128
        blocks = (n + threads_per_block - 1) // threads_per_block
        _initialize_xoroshiro128p_states[blocks, threads_per_block, stream](
            states, np.uint64(seed)
        )
    return states


@cuda.jit(device=True)
def load_local_rng(states, index, local_state):
    """Read the two state words once at the start of a fused chunk."""
    local_state[0] = states[index]["s0"]
    local_state[1] = states[index]["s1"]


@cuda.jit(device=True)
def prepare_local_rng(states, index, seed, first_chunk, local_state):
    """Initialize a new chunk or resume the walker's saved RNG stream."""
    if first_chunk:
        initialize_local_rng(seed, index, local_state)
    else:
        load_local_rng(states, index, local_state)


@cuda.jit(device=True)
def store_local_rng(states, index, local_state):
    """Commit a thread's state after a fused chunk."""
    states[index]["s0"] = local_state[0]
    states[index]["s1"] = local_state[1]


@cuda.jit(device=True)
def _local_next_uint64(local_state):
    s0 = local_state[0]
    s1 = local_state[1]
    result = s0 + s1
    s1 ^= s0
    local_state[0] = (
        ((s0 << np.uint32(55)) | (s0 >> np.uint32(9))) ^ s1 ^ (s1 << np.uint32(14))
    )
    local_state[1] = (s1 << np.uint32(36)) | (s1 >> np.uint32(28))
    return result


@cuda.jit(device=True)
def xoroshiro128p_uniform_local_float64(local_state):
    """Return one uniform draw while updating chunk-local state."""
    value = _local_next_uint64(local_state)
    return np.float64(value >> np.uint32(11)) * (1.0 / 9007199254740992.0)


@cuda.jit(device=True)
def xoroshiro128p_unit_sphere_local_float64(local_state, vector):
    """Match the two-uniform isotropic sphere draw in ``_gpu_random``."""
    z = 2.0 * xoroshiro128p_uniform_local_float64(local_state) - 1.0
    phi = 2.0 * math.pi * xoroshiro128p_uniform_local_float64(local_state)
    radial_squared = 1.0 - z * z
    if radial_squared < 0.0:  # noqa: PLR1730 - preserve existing draw operations
        radial_squared = 0.0
    radial = math.sqrt(radial_squared)
    vector[0] = radial * math.cos(phi)
    vector[1] = radial * math.sin(phi)
    vector[2] = z


# Device position initialization

"""Direct GPU initialization for analytic substrates and uniform voxels."""


@cuda.jit
def _cuda_init_zero(positions):
    walker = cuda.grid(1)
    if walker >= positions.shape[0]:
        return
    for axis in range(3):
        positions[walker, axis] = 0.0


@cuda.jit
def _cuda_init_sphere(positions, init_seed, radius):
    walker = cuda.grid(1)
    if walker >= positions.shape[0]:
        return
    rng_state = cuda.local.array(2, np.uint64)
    initialize_local_rng(init_seed, walker, rng_state)
    direction = cuda.local.array(3, np.float64)
    xoroshiro128p_unit_sphere_local_float64(rng_state, direction)
    distance = radius * math.pow(xoroshiro128p_uniform_local_float64(rng_state), 1 / 3)
    for axis in range(3):
        positions[walker, axis] = distance * direction[axis]


@cuda.jit
def _cuda_init_cylinder(positions, init_seed, radius):
    walker = cuda.grid(1)
    if walker >= positions.shape[0]:
        return
    rng_state = cuda.local.array(2, np.uint64)
    initialize_local_rng(init_seed, walker, rng_state)
    distance = radius * math.sqrt(xoroshiro128p_uniform_local_float64(rng_state))
    angle = 2.0 * math.pi * xoroshiro128p_uniform_local_float64(rng_state)
    positions[walker, 0] = 0.0
    positions[walker, 1] = distance * math.cos(angle)
    positions[walker, 2] = distance * math.sin(angle)


@cuda.jit
def _cuda_init_ellipsoid(positions, init_seed, semiaxes):
    walker = cuda.grid(1)
    if walker >= positions.shape[0]:
        return
    rng_state = cuda.local.array(2, np.uint64)
    initialize_local_rng(init_seed, walker, rng_state)
    direction = cuda.local.array(3, np.float64)
    xoroshiro128p_unit_sphere_local_float64(rng_state, direction)
    radial = math.pow(xoroshiro128p_uniform_local_float64(rng_state), 1 / 3)
    for axis in range(3):
        positions[walker, axis] = radial * direction[axis] * semiaxes[axis]


@cuda.jit
def _cuda_init_uniform_voxel(positions, init_seed, voxel_size):
    walker = cuda.grid(1)
    if walker >= positions.shape[0]:
        return
    rng_state = cuda.local.array(2, np.uint64)
    initialize_local_rng(init_seed, walker, rng_state)
    for axis in range(3):
        positions[walker, axis] = (
            xoroshiro128p_uniform_local_float64(rng_state) * voxel_size[axis]
        )


def _device_initial_positions(
    kind, n_walkers, seed, block_size, stream, *, radius=None, dimensions=None
):
    """Return positions sampled directly on the device.

    Cylinder and ellipsoid coordinates are in their local substrate frames.
    ``dimensions`` is an existing device array containing semiaxes or voxel
    dimensions. The initialization RNG seed uses an independent domain and
    each walker derives its two state words on the GPU, without a state upload.
    """
    blocks = (n_walkers + block_size - 1) // block_size
    init_seed = np.uint64(derive_seed(seed, 2))
    positions = cuda.device_array((n_walkers, 3), dtype=np.float64, stream=stream)
    if kind == "free":
        _cuda_init_zero[blocks, block_size, stream](positions)
    elif kind == "sphere":
        _cuda_init_sphere[blocks, block_size, stream](positions, init_seed, radius)
    elif kind == "cylinder":
        _cuda_init_cylinder[blocks, block_size, stream](positions, init_seed, radius)
    elif kind == "ellipsoid":
        _cuda_init_ellipsoid[blocks, block_size, stream](
            positions, init_seed, dimensions
        )
    elif kind == "uniform_voxel":
        _cuda_init_uniform_voxel[blocks, block_size, stream](
            positions, init_seed, dimensions
        )
    else:
        raise ValueError(f"Unknown initial-position kind: {kind}")
    return positions


# Collision helpers and one-step kernels

"""CUDA device helpers and diffusion simulation kernels.

The host-side stream, transfer, and output logic lives in :mod:`simulations`.
"""


@cuda.jit(device=True)
def _cuda_dot_product(a, b):
    """Calculate the dot product between two 1D arrays of length 3.

    Parameters
    ----------
    a : numba.cuda.cudadrv.devicearray.DeviceNDArray
    b : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    float
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit(device=True)
def _cuda_cross_product(a, b, c):
    """Calculate the cross product between two 1D arrays of length 3.

    Parameters
    ----------
    a : numba.cuda.cudadrv.devicearray.DeviceNDArray
    b : numba.cuda.cudadrv.devicearray.DeviceNDArray
    c : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return


@cuda.jit(device=True)
def _cuda_normalize_vector(v):
    """Scale 1D array of length 3 so that it has unit length.

    Parameters
    ----------
    v : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    length = math.sqrt(_cuda_dot_product(v, v))
    for i in range(3):
        v[i] = v[i] / length
    return


@cuda.jit(device=True)
def _cuda_triangle_normal(triangle, normal):
    """Calculate the normal vector of a triangle.

    Parameters
    ----------
    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray
    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    v = cuda.local.array(3, np.float64)
    k = cuda.local.array(3, np.float64)
    for i in range(3):
        v[i] = triangle[0, i] - triangle[1, i]
        k[i] = triangle[0, i] - triangle[2, i]
    _cuda_cross_product(v, k, normal)
    _cuda_normalize_vector(normal)
    return


@cuda.jit(device=True)
def _cuda_get_triangle(i, vertices, faces, triangle):
    """Get the ith triangle from vertices and faces arrays.

    Parameters
    ----------
    i : int
    vertices : numba.cuda.cudadrv.devicearray.DeviceNDArray
    faces : numba.cuda.cudadrv.devicearray.DeviceNDArray
    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    for a in range(3):
        for b in range(3):
            triangle[a, b] = vertices[faces[i, a], b]
    return


@cuda.jit(device=True)
def _cuda_random_step(step, rng_states, thread_id):
    """Generate a random step from a uniform distribution over a sphere.

    Parameters
    ----------
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    rng_states : numba.cuda.cudadrv.devicearray.DeviceNDArray
    thread_id : int

    Returns
    -------
    None
    """
    xoroshiro128p_unit_sphere_float64(rng_states, thread_id, step)
    return


@cuda.jit(device=True)
def _cuda_mat_mul(R, v):
    """Multiply 1D array v of length 3 by matrix R of size 3 x 3.

    Parameters
    ----------
    R : numba.cuda.cudadrv.devicearray.DeviceNDArray
    v : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    rotated_v = cuda.local.array(3, np.float64)
    rotated_v[0] = R[0, 0] * v[0] + R[0, 1] * v[1] + R[0, 2] * v[2]
    rotated_v[1] = R[1, 0] * v[0] + R[1, 1] * v[1] + R[1, 2] * v[2]
    rotated_v[2] = R[2, 0] * v[0] + R[2, 1] * v[1] + R[2, 2] * v[2]
    for i in range(3):
        v[i] = rotated_v[i]
    return


@cuda.jit(device=True)
def _cuda_line_circle_intersection(r0, step, radius):
    """Calculate the distance from r0 to a circle centered at origin along
    step. r0 must be inside the circle.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    radius : float

    Returns
    -------
    float
    """
    A = step[0] ** 2 + step[1] ** 2
    B = 2 * (r0[0] * step[0] + r0[1] * step[1])
    C = r0[0] ** 2 + r0[1] ** 2 - radius**2
    d = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return d


@cuda.jit(device=True)
def _cuda_line_sphere_intersection(r0, step, radius):
    """Calculate the distance from r0 to a sphere centered at origin along
    step. r0 must be inside the sphere.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    radius : float

    Returns
    -------
    float
    """
    dp = _cuda_dot_product(step, r0)
    d = -dp + math.sqrt(dp**2 - (_cuda_dot_product(r0, r0) - radius**2))
    return d


@cuda.jit(device=True)
def _cuda_line_ellipsoid_intersection(r0, step, semiaxes):
    """Calculate the distance from r0 to an axis-aligned ellipsoid centered at
    origin along step. r0 must be inside the ellipsoid.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    semiaxes : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    float
    """
    a = semiaxes[0]
    b = semiaxes[1]
    c = semiaxes[2]
    A = (step[0] / a) ** 2 + (step[1] / b) ** 2 + (step[2] / c) ** 2
    B = 2 * (
        a ** (-2) * step[0] * r0[0]
        + b ** (-2) * step[1] * r0[1]
        + c ** (-2) * step[2] * r0[2]
    )
    C = (r0[0] / a) ** 2 + (r0[1] / b) ** 2 + (r0[2] / c) ** 2 - 1
    d = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return d


@cuda.jit(device=True)
def _cuda_ray_triangle_intersection_check(triangle, r0, step):
    """Check if a ray defined by r0 and step intersects with a triangle defined
    by A, B, and C. The output is the distance in units of step length from r0
    to intersection if intersection found, nan otherwise. This function is
    based on the Moller-Trumbore algorithm.

    Parameters
    ----------
    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    float or nan
    """
    A = triangle[0]
    B = triangle[1]
    C = triangle[2]
    T = cuda.local.array(3, np.float64)
    E_1 = cuda.local.array(3, np.float64)
    E_2 = cuda.local.array(3, np.float64)
    P = cuda.local.array(3, np.float64)
    Q = cuda.local.array(3, np.float64)
    for i in range(3):
        T[i] = r0[i] - A[i]
        E_1[i] = B[i] - A[i]
        E_2[i] = C[i] - A[i]
    _cuda_cross_product(step, E_2, P)
    _cuda_cross_product(T, E_1, Q)
    det = _cuda_dot_product(P, E_1)
    if det != 0:
        t = 1 / det * _cuda_dot_product(Q, E_2)
        u = 1 / det * _cuda_dot_product(P, T)
        v = 1 / det * _cuda_dot_product(Q, step)
        if u >= 0 and u <= 1 and v >= 0 and v <= 1 and u + v <= 1:
            return t
        else:
            return np.nan
    else:
        return np.nan


@cuda.jit(device=True)
def _cuda_ray_triangle_intersection_prepacked(triangle_data, index, r0, step):
    """Moller-Trumbore intersection using prepacked origin and edge vectors."""
    ax = triangle_data[index, 0]
    ay = triangle_data[index, 1]
    az = triangle_data[index, 2]
    e1x = triangle_data[index, 3]
    e1y = triangle_data[index, 4]
    e1z = triangle_data[index, 5]
    e2x = triangle_data[index, 6]
    e2y = triangle_data[index, 7]
    e2z = triangle_data[index, 8]

    px = step[1] * e2z - step[2] * e2y
    py = step[2] * e2x - step[0] * e2z
    pz = step[0] * e2y - step[1] * e2x
    determinant = px * e1x + py * e1y + pz * e1z
    if determinant == 0:
        return np.nan

    tx = r0[0] - ax
    ty = r0[1] - ay
    tz = r0[2] - az
    inverse_determinant = 1 / determinant
    u = (px * tx + py * ty + pz * tz) * inverse_determinant
    if u < 0 or u > 1:
        return np.nan

    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (qx * step[0] + qy * step[1] + qz * step[2]) * inverse_determinant
    if v < 0 or u + v > 1:
        return np.nan
    return (qx * e2x + qy * e2y + qz * e2z) * inverse_determinant


@cuda.jit(device=True)
def _cuda_reflection(r0, step, d, normal, epsilon):
    """Calculate reflection and update r0 and step. Epsilon is the amount by
    which the new position differs from the reflection point.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    d : float
    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray
    semiaxes : numba.cuda.cudadrv.devicearray.DeviceNDArray
    epsilon : float

    Returns
    -------
    float
    """
    intersection = cuda.local.array(3, np.float64)
    v = cuda.local.array(3, np.float64)
    for i in range(3):
        intersection[i] = r0[i] + d * step[i]
        v[i] = intersection[i] - r0[i]
    dp = _cuda_dot_product(v, normal)
    if dp > 0:  # Make sure the normal vector points against the step
        for i in range(3):
            normal[i] *= -1
        dp = _cuda_dot_product(v, normal)
    for i in range(3):
        step[i] = (v[i] - 2 * dp * normal[i] + intersection[i]) - intersection[i]
    _cuda_normalize_vector(step)
    for i in range(3):  # Move walker slightly away from the surface
        r0[i] = intersection[i] + epsilon * normal[i]
    return


@cuda.jit(device=True)
def _cuda_crossing(r0, step, d, normal, epsilon):
    """Make the random walker pass through the triangle and update r0. Epsilon
    is the amount by which the new position differs from the surface.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    d : float
    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray
    semiaxes : numba.cuda.cudadrv.devicearray.DeviceNDArray
    epsilon : float

    Returns
    -------
    None
    """
    intersection = cuda.local.array(3, np.float64)
    v = cuda.local.array(3, np.float64)
    for i in range(3):
        intersection[i] = r0[i] + d * step[i]
        v[i] = intersection[i] - r0[i]
    dp = _cuda_dot_product(v, normal)
    if dp < 0:  # Make sure the normal vector points to the other side of the triangle
        for i in range(3):
            normal[i] *= -1
    for i in range(3):  # Move walker slightly away from the surface
        r0[i] = intersection[i] + epsilon * normal[i]
    return


@cuda.jit(device=True)
def _cuda_resolve_mesh_collision(r0, step, d, normal, epsilon, crosses):
    """Resolve one mesh hit and return all path length it consumes."""
    if crosses:
        _cuda_crossing(r0, step, d, normal, epsilon)
    else:
        _cuda_reflection(r0, step, d, normal, epsilon)
    return d + epsilon


@cuda.jit(device=True)
def _cuda_mesh_hit_reflects(triangle_index, n_membrane_faces, perm_prob, uniform_draw):
    """Return the collision decision with exact permeability endpoints."""
    if triangle_index >= n_membrane_faces or perm_prob <= 0:
        return True
    if perm_prob >= 1:
        return False
    return uniform_draw >= perm_prob


@cuda.jit(device=True)
def _cuda_phase_increment(g_x, g_y, g_z, channel, t, x, y, z, dt):
    """Accumulate one phase contribution with the documented trapezoid rule."""
    return (
        GAMMA
        * dt
        * 0.5
        * (
            (g_x[channel, t] + g_x[channel, t + 1]) * x
            + (g_y[channel, t] + g_y[channel, t + 1]) * y
            + (g_z[channel, t] + g_z[channel, t + 1]) * z
        )
    )


@cuda.jit(device=True)
def _cuda_advance_sphere(
    r0, step, length, radius, failures, thread_id, max_iter, epsilon
):
    """Resolve sphere reflections and write the complete step into ``r0``."""
    iter_idx = 0
    check_intersection = True
    while check_intersection and length > 0 and iter_idx < max_iter:
        iter_idx += 1
        d = _cuda_line_sphere_intersection(r0, step, radius)
        if d > 0 and d < length:
            normal = cuda.local.array(3, np.float64)
            for axis in range(3):
                normal[axis] = -(r0[axis] + d * step[axis])
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal, epsilon)
            length -= d + epsilon
        else:
            check_intersection = False
    if check_intersection and length > 0:
        failures[thread_id] = True
    for axis in range(3):
        r0[axis] = r0[axis] + step[axis] * length


@cuda.jit(device=True)
def _cuda_advance_cylinder(
    r0, step, length, radius, failures, thread_id, max_iter, epsilon
):
    """Resolve cylinder reflections in the cylinder's local frame."""
    iter_idx = 0
    check_intersection = True
    while check_intersection and length > 0 and iter_idx < max_iter:
        iter_idx += 1
        d = _cuda_line_circle_intersection(r0[1:3], step[1:3], radius)
        if d > 0 and d < length:
            normal = cuda.local.array(3, np.float64)
            normal[0] = 0.0
            for axis in range(1, 3):
                normal[axis] = -(r0[axis] + d * step[axis])
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal, epsilon)
            length -= d + epsilon
        else:
            check_intersection = False
    if check_intersection and length > 0:
        failures[thread_id] = True
    for axis in range(3):
        r0[axis] = r0[axis] + step[axis] * length


@cuda.jit(device=True)
def _cuda_advance_ellipsoid(
    r0, step, length, semiaxes, failures, thread_id, max_iter, epsilon
):
    """Resolve ellipsoid reflections in the ellipsoid's local frame."""
    iter_idx = 0
    check_intersection = True
    while check_intersection and length > 0 and iter_idx < max_iter:
        iter_idx += 1
        d = _cuda_line_ellipsoid_intersection(r0, step, semiaxes)
        if d > 0 and d < length:
            normal = cuda.local.array(3, np.float64)
            for axis in range(3):
                normal[axis] = -(r0[axis] + d * step[axis]) / semiaxes[axis] ** 2
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal, epsilon)
            length -= d + epsilon
        else:
            check_intersection = False
    if check_intersection and length > 0:
        failures[thread_id] = True
    for axis in range(3):
        r0[axis] = r0[axis] + step[axis] * length


@cuda.jit(device=True)
def _cuda_reflect_outward_voxel_start(position, step, xs, ys, zs):
    """Reflect an outward step that begins exactly on an impermeable wall.

    The triangle intersection search excludes zero-distance hits to avoid
    re-hitting a surface immediately after a collision. A user-supplied
    position may nevertheless start *on* a voxel wall, where the outward
    component must reflect before the search begins.
    """
    if (position[0] == xs[0] and step[0] < 0) or (
        position[0] == xs[-1] and step[0] > 0
    ):
        step[0] = -step[0]
    if (position[1] == ys[0] and step[1] < 0) or (
        position[1] == ys[-1] and step[1] > 0
    ):
        step[1] = -step[1]
    if (position[2] == zs[0] and step[2] < 0) or (
        position[2] == zs[-1] and step[2] > 0
    ):
        step[2] = -step[2]


@cuda.jit(device=True)
def _cuda_mesh_inside(
    point,
    triangle_data,
    voxel_size,
    triangle_indices,
    subvoxel_indices,
    xs,
    ys,
    zs,
    n_sv,
    n_faces,
):
    """Classify a point by counting ordered surface crossings along +x.

    Repeated closest-hit queries count a shared-edge or duplicated CSR hit only
    once after advancing beyond that surface.  ``n_faces + 1`` is a geometric
    termination bound; exceeding it signals malformed geometry or numerical
    failure without a fixed-size triangle-ID workspace.
    """
    ray = cuda.local.array(3, np.float64)
    ray[0] = 1.0
    ray[1] = 0.0
    ray[2] = 0.0
    origin = cuda.local.array(3, np.float64)
    for axis in range(3):
        origin[axis] = point[axis]
    closest_triangle_index = cuda.local.array(1, np.int64)
    coordinate_scale = max(voxel_size[0], max(voxel_size[1], voxel_size[2]))
    surface_offset = 7.105427357601002e-15 * coordinate_scale
    # Include a membrane closing exactly on the upper voxel face. The
    # classification CSR excludes lower-face membranes so their periodic
    # copies at this endpoint cannot introduce a spurious crossing.
    remaining = voxel_size[0] - point[0] + 4 * surface_offset
    n_intersections = 0
    for _ in range(n_faces + 1):
        distance = _cuda_find_mesh_intersection(
            origin,
            ray,
            remaining,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            closest_triangle_index,
        )
        if closest_triangle_index[0] < 0 or distance > remaining:
            return n_intersections % 2
        n_intersections += 1
        advance = distance + surface_offset
        origin[0] += advance
        remaining -= advance
        if remaining <= 0:
            return n_intersections % 2
    return -1


@cuda.jit()
def _cuda_copy_rng_states(source, destination):
    """Preserve advanced streams when a sparse mesh batch grows."""
    index = cuda.grid(1)
    if index < source.shape[0]:
        destination[index]["s0"] = source[index]["s0"]
        destination[index]["s1"] = source[index]["s1"]


@cuda.jit()
def _cuda_fill_mesh(
    candidates,
    block_counts,
    classifier_failures,
    rng_states,
    intra,
    triangle_data,
    voxel_size,
    triangle_indices,
    subvoxel_indices,
    xs,
    ys,
    zs,
    n_sv,
    n_faces,
):
    """Classify one candidate per thread and compact within each block."""
    thread_id = cuda.grid(1)
    lane = cuda.threadIdx.x
    accepted_scan = cuda.shared.array(1024, np.int32)
    point = cuda.local.array(3, np.float64)
    accepted = 0
    if thread_id < candidates.shape[0]:
        for i in range(3):
            point[i] = (
                xoroshiro128p_uniform_float64(rng_states, thread_id) * voxel_size[i]
            )

        inside = _cuda_mesh_inside(
            point,
            triangle_data,
            voxel_size,
            triangle_indices,
            subvoxel_indices,
            xs,
            ys,
            zs,
            n_sv,
            n_faces,
        )
        if inside < 0:
            cuda.atomic.add(classifier_failures, 0, 1)
        elif (intra and inside == 1) or (not intra and inside == 0):
            accepted = 1

    # Inclusive Hillis-Steele scan. Every thread participates, including the
    # inactive lanes of the final partial block, so synchronization is valid.
    accepted_scan[lane] = accepted
    cuda.syncthreads()
    offset = 1
    while offset < cuda.blockDim.x:
        previous = 0
        if lane >= offset:
            previous = accepted_scan[lane - offset]
        cuda.syncthreads()
        if lane >= offset:
            accepted_scan[lane] += previous
        cuda.syncthreads()
        offset *= 2

    if accepted == 1:
        candidate_index = cuda.blockIdx.x * cuda.blockDim.x + accepted_scan[lane] - 1
        for axis in range(3):
            candidates[candidate_index, axis] = point[axis]
    if lane == cuda.blockDim.x - 1:
        block_counts[cuda.blockIdx.x] = accepted_scan[lane]
    return


@cuda.jit()
def _cuda_prefix_mesh_block_counts(block_counts, block_offsets, batch_accepted):
    """Build deterministic block offsets and a scalar accepted count on device."""
    if cuda.grid(1) != 0:
        return
    offset = 0
    for block in range(block_counts.shape[0]):
        block_offsets[block] = offset
        offset += block_counts[block]
    batch_accepted[0] = offset


@cuda.jit()
def _cuda_compact_mesh_candidates(
    candidates, block_counts, block_offsets, accepted_before, points
):
    """Copy block-compacted candidates into deterministic global slots."""
    block = cuda.blockIdx.x
    lane = cuda.threadIdx.x
    if block >= block_counts.shape[0] or lane >= block_counts[block]:
        return
    output_index = accepted_before + block_offsets[block] + lane
    if output_index >= points.shape[0]:
        return
    candidate_index = block * cuda.blockDim.x + lane
    for axis in range(3):
        points[output_index, axis] = candidates[candidate_index, axis]
    return


@cuda.jit(device=True)
def _ll_subvoxel_overlap(xs, x1, x2):
    """Return the lower half-open index for a closed interval and grid."""
    xmin = min(x1, x2)
    n_subvoxels = len(xs) - 1
    if xmin < xs[0]:
        return 0
    elif xmin > xs[-1]:
        return n_subvoxels
    elif xmin == xs[-1]:
        return n_subvoxels - 1
    else:
        for i in range(1, len(xs)):
            if xs[i] > xmin:
                return i - 1
        return n_subvoxels - 1


@cuda.jit(device=True)
def _ul_subvoxel_overlap(xs, x1, x2):
    """Return the upper half-open index for a closed interval and grid."""
    xmax = max(x1, x2)
    if xmax > xs[-1]:
        return len(xs) - 1
    elif xmax < xs[0]:
        return 0
    else:
        for i in range(1, len(xs)):
            if xs[i] > xmax:
                return i
        return len(xs) - 1


@cuda.jit(device=True)
def _ll_subvoxel_overlap_periodic(xs, x1, x2):
    """For an interval [x1, x2], return the index of the lower limit of the
    overlapping subvoxels whose borders are defined by the elements of xs, and
    the division continues periodically."""
    xmin = min(x1, x2)
    voxel_size = xs[-1] - xs[0]
    n_subvoxels = len(xs) - 1
    n = math.floor((xmin - xs[0]) / voxel_size)
    xmin_shifted = xmin - n * voxel_size
    ll_shifted = _ll_subvoxel_overlap(xs, xmin_shifted, xmin_shifted)
    ll = ll_shifted + n * n_subvoxels
    # A periodic seam belongs to both adjacent copies.  Mesh triangles on the
    # upper voxel boundary are indexed in the preceding copy, while triangles
    # on the lower boundary are indexed in the following copy.
    if xmin_shifted == xs[0]:
        ll -= 1
    return ll


@cuda.jit(device=True)
def _ul_subvoxel_overlap_periodic(xs, x1, x2):
    """For an interval [x1, x2], return the index of the upper limit of the
    overlapping subvoxels whose borders are defined by the elements of xs, and
    the division continues periodically."""
    xmax = max(x1, x2)
    voxel_size = xs[-1] - xs[0]
    n_subvoxels = len(xs) - 1
    n = math.floor((xmax - xs[0]) / voxel_size)
    xmax_shifted = xmax - n * voxel_size
    ul_shifted = _ul_subvoxel_overlap(xs, xmax_shifted, xmax_shifted)
    ul = ul_shifted + n * n_subvoxels
    return ul


@cuda.jit()
def _cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, t, step_l, dt):
    """Kernel function for free diffusion."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, np.float64)
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, np.uint64(0), False, state)
    xoroshiro128p_unit_sphere_local_float64(state, step)
    for i in range(3):
        positions[thread_id, i] = positions[thread_id, i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += _cuda_phase_increment(
            g_x,
            g_y,
            g_z,
            m,
            t,
            positions[thread_id, 0],
            positions[thread_id, 1],
            positions[thread_id, 2],
            dt,
        )
    store_local_rng(rng_states, thread_id, state)
    return


@cuda.jit()
def _cuda_step_sphere(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    radius,
    iter_exc,
    max_iter,
    epsilon,
):
    """Kernel function for diffusion inside a sphere."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, np.float64)
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, np.uint64(0), False, state)
    xoroshiro128p_unit_sphere_local_float64(state, step)
    r0 = positions[thread_id, :]
    _cuda_advance_sphere(
        r0, step, step_l, radius, iter_exc, thread_id, max_iter, epsilon
    )
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += _cuda_phase_increment(
            g_x,
            g_y,
            g_z,
            m,
            t,
            positions[thread_id, 0],
            positions[thread_id, 1],
            positions[thread_id, 2],
            dt,
        )
    store_local_rng(rng_states, thread_id, state)
    return


@cuda.jit(device=True)
def _cuda_check_mesh_cell(
    r0,
    step,
    step_l,
    triangle_data,
    xs,
    ys,
    zs,
    subvoxel_indices,
    triangle_indices,
    n_sv,
    x_unwrapped,
    y_unwrapped,
    z_unwrapped,
    min_d,
    closest_triangle_index,
):
    """Packed-geometry variant of the periodic-cell intersection search."""
    temp_r0 = cuda.local.array(3, np.float64)
    nx = len(xs) - 1
    ny = len(ys) - 1
    nz = len(zs) - 1
    x_shift_n = math.floor(x_unwrapped / nx)
    y_shift_n = math.floor(y_unwrapped / ny)
    z_shift_n = math.floor(z_unwrapped / nz)
    x = x_unwrapped - x_shift_n * nx
    y = y_unwrapped - y_shift_n * ny
    z = z_unwrapped - z_shift_n * nz
    temp_r0[0] = r0[0] - x_shift_n * (xs[-1] - xs[0])
    temp_r0[1] = r0[1] - y_shift_n * (ys[-1] - ys[0])
    temp_r0[2] = r0[2] - z_shift_n * (zs[-1] - zs[0])
    sv = int(x * n_sv[1] * n_sv[2] + y * n_sv[2] + z)

    for i in range(subvoxel_indices[sv, 0], subvoxel_indices[sv, 1]):
        triangle_index = triangle_indices[i]
        d = _cuda_ray_triangle_intersection_prepacked(
            triangle_data, triangle_index, temp_r0, step
        )
        if d > 0 and d <= step_l and d < min_d:
            closest_triangle_index[0] = triangle_index
            min_d = d
    return min_d


@cuda.jit(device=True)
def _cuda_start_subvoxel(x, direction, grid_origin, subvoxel_size):
    """Return the unwrapped cell entered by a ray at ``x``."""
    cell = math.floor((x - grid_origin) / subvoxel_size)
    if direction < 0 and x == grid_origin + cell * subvoxel_size:
        cell -= 1
    return cell


@cuda.jit(device=True)
def _cuda_zero_direction_voxel_seam(x, direction, grid_origin, voxel_size):
    """Return whether a stationary coordinate lies on a periodic seam."""
    if direction != 0:
        return False
    copy_index = math.floor((x - grid_origin) / voxel_size)
    return x == grid_origin + copy_index * voxel_size


@cuda.jit(device=True)
def _cuda_find_mesh_intersection(
    r0,
    step,
    step_l,
    triangle_data,
    xs,
    ys,
    zs,
    subvoxel_indices,
    triangle_indices,
    n_sv,
    closest_triangle_index,
):
    """Find the closest mesh hit with ordered regular-grid DDA traversal."""
    dx = (xs[-1] - xs[0]) / (len(xs) - 1)
    dy = (ys[-1] - ys[0]) / (len(ys) - 1)
    dz = (zs[-1] - zs[0]) / (len(zs) - 1)
    x = _cuda_start_subvoxel(r0[0], step[0], xs[0], dx)
    y = _cuda_start_subvoxel(r0[1], step[1], ys[0], dy)
    z = _cuda_start_subvoxel(r0[2], step[2], zs[0], dz)

    step_x = 0
    step_y = 0
    step_z = 0
    t_max_x = math.inf
    t_max_y = math.inf
    t_max_z = math.inf
    t_delta_x = math.inf
    t_delta_y = math.inf
    t_delta_z = math.inf
    if step[0] > 0:
        step_x = 1
        t_max_x = (xs[0] + (x + 1) * dx - r0[0]) / step[0]
        t_delta_x = dx / step[0]
    elif step[0] < 0:
        step_x = -1
        t_max_x = (xs[0] + x * dx - r0[0]) / step[0]
        t_delta_x = -dx / step[0]
    if step[1] > 0:
        step_y = 1
        t_max_y = (ys[0] + (y + 1) * dy - r0[1]) / step[1]
        t_delta_y = dy / step[1]
    elif step[1] < 0:
        step_y = -1
        t_max_y = (ys[0] + y * dy - r0[1]) / step[1]
        t_delta_y = -dy / step[1]
    if step[2] > 0:
        step_z = 1
        t_max_z = (zs[0] + (z + 1) * dz - r0[2]) / step[2]
        t_delta_z = dz / step[2]
    elif step[2] < 0:
        step_z = -1
        t_max_z = (zs[0] + z * dz - r0[2]) / step[2]
        t_delta_z = -dz / step[2]

    closest_triangle_index[0] = -1
    min_d = _cuda_check_mesh_cell(
        r0,
        step,
        step_l,
        triangle_data,
        xs,
        ys,
        zs,
        subvoxel_indices,
        triangle_indices,
        n_sv,
        x,
        y,
        z,
        math.inf,
        closest_triangle_index,
    )

    # A ray lying on a periodic seam can intersect triangles stored on either
    # side of that seam.  Inspect every adjacent seam cell once at the start.
    seam_x = _cuda_zero_direction_voxel_seam(r0[0], step[0], xs[0], xs[-1] - xs[0])
    seam_y = _cuda_zero_direction_voxel_seam(r0[1], step[1], ys[0], ys[-1] - ys[0])
    seam_z = _cuda_zero_direction_voxel_seam(r0[2], step[2], zs[0], zs[-1] - zs[0])
    if seam_x:
        min_d = _cuda_check_mesh_cell(
            r0,
            step,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            x - 1,
            y,
            z,
            min_d,
            closest_triangle_index,
        )
    if seam_y:
        min_d = _cuda_check_mesh_cell(
            r0,
            step,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            x,
            y - 1,
            z,
            min_d,
            closest_triangle_index,
        )
    if seam_z:
        min_d = _cuda_check_mesh_cell(
            r0,
            step,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            x,
            y,
            z - 1,
            min_d,
            closest_triangle_index,
        )
    if seam_x and seam_y:
        min_d = _cuda_check_mesh_cell(
            r0,
            step,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            x - 1,
            y - 1,
            z,
            min_d,
            closest_triangle_index,
        )
    if seam_x and seam_z:
        min_d = _cuda_check_mesh_cell(
            r0,
            step,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            x - 1,
            y,
            z - 1,
            min_d,
            closest_triangle_index,
        )
    if seam_y and seam_z:
        min_d = _cuda_check_mesh_cell(
            r0,
            step,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            x,
            y - 1,
            z - 1,
            min_d,
            closest_triangle_index,
        )
    if seam_x and seam_y and seam_z:
        min_d = _cuda_check_mesh_cell(
            r0,
            step,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            x - 1,
            y - 1,
            z - 1,
            min_d,
            closest_triangle_index,
        )

    while True:
        # The MLIR CUDA backend only lowers the two-argument ``min`` form.
        next_t = min(t_max_x, min(t_max_y, t_max_z))
        if next_t > step_l or next_t > min_d:
            break
        tie_tolerance = 3.552713678800501e-15 * max(1.0, abs(next_t))
        cross_x = abs(t_max_x - next_t) <= tie_tolerance
        cross_y = abs(t_max_y - next_t) <= tie_tolerance
        cross_z = abs(t_max_z - next_t) <= tie_tolerance

        # At an exact edge or corner, a triangle touching the crossing can be
        # indexed in any adjacent cell.  Check all nonempty combinations.
        if cross_x:
            min_d = _cuda_check_mesh_cell(
                r0,
                step,
                step_l,
                triangle_data,
                xs,
                ys,
                zs,
                subvoxel_indices,
                triangle_indices,
                n_sv,
                x + step_x,
                y,
                z,
                min_d,
                closest_triangle_index,
            )
        if cross_y:
            min_d = _cuda_check_mesh_cell(
                r0,
                step,
                step_l,
                triangle_data,
                xs,
                ys,
                zs,
                subvoxel_indices,
                triangle_indices,
                n_sv,
                x,
                y + step_y,
                z,
                min_d,
                closest_triangle_index,
            )
        if cross_z:
            min_d = _cuda_check_mesh_cell(
                r0,
                step,
                step_l,
                triangle_data,
                xs,
                ys,
                zs,
                subvoxel_indices,
                triangle_indices,
                n_sv,
                x,
                y,
                z + step_z,
                min_d,
                closest_triangle_index,
            )
        if cross_x and cross_y:
            min_d = _cuda_check_mesh_cell(
                r0,
                step,
                step_l,
                triangle_data,
                xs,
                ys,
                zs,
                subvoxel_indices,
                triangle_indices,
                n_sv,
                x + step_x,
                y + step_y,
                z,
                min_d,
                closest_triangle_index,
            )
        if cross_x and cross_z:
            min_d = _cuda_check_mesh_cell(
                r0,
                step,
                step_l,
                triangle_data,
                xs,
                ys,
                zs,
                subvoxel_indices,
                triangle_indices,
                n_sv,
                x + step_x,
                y,
                z + step_z,
                min_d,
                closest_triangle_index,
            )
        if cross_y and cross_z:
            min_d = _cuda_check_mesh_cell(
                r0,
                step,
                step_l,
                triangle_data,
                xs,
                ys,
                zs,
                subvoxel_indices,
                triangle_indices,
                n_sv,
                x,
                y + step_y,
                z + step_z,
                min_d,
                closest_triangle_index,
            )
        if cross_x and cross_y and cross_z:
            min_d = _cuda_check_mesh_cell(
                r0,
                step,
                step_l,
                triangle_data,
                xs,
                ys,
                zs,
                subvoxel_indices,
                triangle_indices,
                n_sv,
                x + step_x,
                y + step_y,
                z + step_z,
                min_d,
                closest_triangle_index,
            )

        if cross_x:
            x += step_x
            t_max_x += t_delta_x
        if cross_y:
            y += step_y
            t_max_y += t_delta_y
        if cross_z:
            z += step_z
            t_max_z += t_delta_z

    return min_d


@cuda.jit(device=True)
def _cuda_advance_mesh(
    r0,
    step,
    state,
    length,
    triangle_data,
    xs,
    ys,
    zs,
    subvoxel_indices,
    triangle_indices,
    failures,
    thread_id,
    max_iter,
    n_sv,
    epsilon,
    perm_prob,
    n_membrane_faces,
):
    """Resolve a mesh step using the walker's local RNG stream."""
    if n_membrane_faces < triangle_data.shape[0]:
        _cuda_reflect_outward_voxel_start(r0, step, xs, ys, zs)
    normal = cuda.local.array(3, np.float64)
    closest_triangle_index = cuda.local.array(1, np.int64)
    check_intersection = True
    iter_idx = 0
    while check_intersection and length > 0 and iter_idx < max_iter:
        iter_idx += 1
        min_d = _cuda_find_mesh_intersection(
            r0,
            step,
            length,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            n_sv,
            closest_triangle_index,
        )
        if min_d > length:
            check_intersection = False
        else:
            uniform_draw = 0.0
            if closest_triangle_index[0] < n_membrane_faces:
                uniform_draw = xoroshiro128p_uniform_local_float64(state)
            reflects = _cuda_mesh_hit_reflects(
                closest_triangle_index[0], n_membrane_faces, perm_prob, uniform_draw
            )
            for axis in range(3):
                normal[axis] = triangle_data[closest_triangle_index[0], 9 + axis]
            length -= _cuda_resolve_mesh_collision(
                r0, step, min_d, normal, epsilon, not reflects
            )
    if check_intersection and length > 0:
        failures[thread_id] = True
    for axis in range(3):
        r0[axis] = r0[axis] + step[axis] * length


@cuda.jit()
def _cuda_step_mesh(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    triangle_data,
    xs,
    ys,
    zs,
    subvoxel_indices,
    triangle_indices,
    iter_exc,
    max_iter,
    n_sv,
    epsilon,
    perm_prob,
    n_membrane_faces,
):
    """Kernel function for diffusion restricted by a triangular mesh."""

    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return

    step = cuda.local.array(3, np.float64)
    r0 = positions[thread_id, :]
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, np.uint64(0), False, state)
    xoroshiro128p_unit_sphere_local_float64(state, step)
    _cuda_advance_mesh(
        r0,
        step,
        state,
        step_l,
        triangle_data,
        xs,
        ys,
        zs,
        subvoxel_indices,
        triangle_indices,
        iter_exc,
        thread_id,
        max_iter,
        n_sv,
        epsilon,
        perm_prob,
        n_membrane_faces,
    )
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += _cuda_phase_increment(
            g_x,
            g_y,
            g_z,
            m,
            t,
            positions[thread_id, 0],
            positions[thread_id, 1],
            positions[thread_id, 2],
            dt,
        )
    store_local_rng(rng_states, thread_id, state)
    return


# Fused diffusion kernels

"""Experimental fused diffusion kernels for measured GPU launch selection.

The host selects a chunk length. Each thread keeps its position and RNG state
until the chunk ends; small phase bases also stay in per-thread local storage.
The one-step kernels remain available for trajectory output and comparison.
"""


@cuda.jit()
def _cuda_fused_free_small_phase(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    start,
    stop,
    seed,
    first_chunk,
    step_l,
    dt,
):
    """Fuse free steps while retaining up to 16 phase channels locally."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, seed, first_chunk, state)
    position = cuda.local.array(3, np.float64)
    step = cuda.local.array(3, np.float64)
    local_phases = cuda.local.array(16, np.float64)
    for axis in range(3):
        position[axis] = positions[thread_id, axis]
    for channel in range(g_x.shape[0]):
        local_phases[channel] = phases[channel, thread_id]

    for t in range(start, stop):
        xoroshiro128p_unit_sphere_local_float64(state, step)
        for axis in range(3):
            position[axis] += step[axis] * step_l
        for channel in range(g_x.shape[0]):
            local_phases[channel] += _cuda_phase_increment(
                g_x, g_y, g_z, channel, t, position[0], position[1], position[2], dt
            )

    for axis in range(3):
        positions[thread_id, axis] = position[axis]
    for channel in range(g_x.shape[0]):
        phases[channel, thread_id] = local_phases[channel]
    store_local_rng(rng_states, thread_id, state)


@cuda.jit()
def _cuda_step_cylinder_local(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    radius,
    iter_exc,
    max_iter,
    epsilon,
):
    """One cylinder step with position and gradient in the cylinder frame."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, np.float64)
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, np.uint64(0), False, state)
    xoroshiro128p_unit_sphere_local_float64(state, step)
    r0 = positions[thread_id, :]
    _cuda_advance_cylinder(
        r0, step, step_l, radius, iter_exc, thread_id, max_iter, epsilon
    )
    for channel in range(g_x.shape[0]):
        phases[channel, thread_id] += _cuda_phase_increment(
            g_x,
            g_y,
            g_z,
            channel,
            t,
            positions[thread_id, 0],
            positions[thread_id, 1],
            positions[thread_id, 2],
            dt,
        )
    store_local_rng(rng_states, thread_id, state)


@cuda.jit()
def _cuda_step_ellipsoid_local(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    semiaxes,
    iter_exc,
    max_iter,
    epsilon,
):
    """One ellipsoid step with position and gradient in its local frame."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, np.float64)
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, np.uint64(0), False, state)
    xoroshiro128p_unit_sphere_local_float64(state, step)
    r0 = positions[thread_id, :]
    _cuda_advance_ellipsoid(
        r0, step, step_l, semiaxes, iter_exc, thread_id, max_iter, epsilon
    )
    for channel in range(g_x.shape[0]):
        phases[channel, thread_id] += _cuda_phase_increment(
            g_x,
            g_y,
            g_z,
            channel,
            t,
            positions[thread_id, 0],
            positions[thread_id, 1],
            positions[thread_id, 2],
            dt,
        )
    store_local_rng(rng_states, thread_id, state)


@cuda.jit()
def _cuda_fused_free_large_phase(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    start,
    stop,
    seed,
    first_chunk,
    step_l,
    dt,
):
    """Fuse free steps with arbitrary phase rank and per-step phase storage."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, seed, first_chunk, state)
    position = cuda.local.array(3, np.float64)
    step = cuda.local.array(3, np.float64)
    for axis in range(3):
        position[axis] = positions[thread_id, axis]

    for t in range(start, stop):
        xoroshiro128p_unit_sphere_local_float64(state, step)
        for axis in range(3):
            position[axis] += step[axis] * step_l
        for channel in range(g_x.shape[0]):
            phases[channel, thread_id] += _cuda_phase_increment(
                g_x, g_y, g_z, channel, t, position[0], position[1], position[2], dt
            )

    for axis in range(3):
        positions[thread_id, axis] = position[axis]
    store_local_rng(rng_states, thread_id, state)


@cuda.jit()
def _cuda_fused_sphere_small_phase(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    start,
    stop,
    seed,
    first_chunk,
    step_l,
    dt,
    radius,
    iter_exc,
    max_iter,
    epsilon,
):
    """Fuse confined sphere steps with up to 16 local phase channels."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, seed, first_chunk, state)
    position = cuda.local.array(3, np.float64)
    step = cuda.local.array(3, np.float64)
    local_phases = cuda.local.array(16, np.float64)
    for axis in range(3):
        position[axis] = positions[thread_id, axis]
    for channel in range(g_x.shape[0]):
        local_phases[channel] = phases[channel, thread_id]

    for t in range(start, stop):
        xoroshiro128p_unit_sphere_local_float64(state, step)
        r0 = cuda.local.array(3, np.float64)
        for axis in range(3):
            r0[axis] = position[axis]
        _cuda_advance_sphere(
            r0, step, step_l, radius, iter_exc, thread_id, max_iter, epsilon
        )
        for axis in range(3):
            position[axis] = r0[axis]
        for channel in range(g_x.shape[0]):
            local_phases[channel] += _cuda_phase_increment(
                g_x, g_y, g_z, channel, t, position[0], position[1], position[2], dt
            )

    for axis in range(3):
        positions[thread_id, axis] = position[axis]
    for channel in range(g_x.shape[0]):
        phases[channel, thread_id] = local_phases[channel]
    store_local_rng(rng_states, thread_id, state)


@cuda.jit()
def _cuda_fused_local_analytic_small_phase(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    start,
    stop,
    seed,
    first_chunk,
    step_l,
    dt,
    kind,
    radius,
    semiaxes,
    iter_exc,
    max_iter,
    epsilon,
):
    """Fuse local cylinder (1) or ellipsoid (2) steps for small phase rank."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, seed, first_chunk, state)
    position = cuda.local.array(3, np.float64)
    step = cuda.local.array(3, np.float64)
    local_phases = cuda.local.array(16, np.float64)
    for axis in range(3):
        position[axis] = positions[thread_id, axis]
    for channel in range(g_x.shape[0]):
        local_phases[channel] = phases[channel, thread_id]

    for t in range(start, stop):
        xoroshiro128p_unit_sphere_local_float64(state, step)
        if kind == 1:
            _cuda_advance_cylinder(
                position, step, step_l, radius, iter_exc, thread_id, max_iter, epsilon
            )
        else:
            _cuda_advance_ellipsoid(
                position, step, step_l, semiaxes, iter_exc, thread_id, max_iter, epsilon
            )
        for channel in range(g_x.shape[0]):
            local_phases[channel] += _cuda_phase_increment(
                g_x, g_y, g_z, channel, t, position[0], position[1], position[2], dt
            )

    for axis in range(3):
        positions[thread_id, axis] = position[axis]
    for channel in range(g_x.shape[0]):
        phases[channel, thread_id] = local_phases[channel]
    store_local_rng(rng_states, thread_id, state)


@cuda.jit()
def _cuda_fused_mesh_small_phase(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    start,
    stop,
    seed,
    first_chunk,
    step_l,
    dt,
    triangle_data,
    xs,
    ys,
    zs,
    subvoxel_indices,
    triangle_indices,
    iter_exc,
    max_iter,
    n_sv,
    epsilon,
    perm_prob,
    n_membrane_faces,
):
    """Fuse mesh steps with cached walker state for a bounded phase rank."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    state = cuda.local.array(2, np.uint64)
    prepare_local_rng(rng_states, thread_id, seed, first_chunk, state)
    position = cuda.local.array(3, np.float64)
    step = cuda.local.array(3, np.float64)
    local_phases = cuda.local.array(16, np.float64)
    for axis in range(3):
        position[axis] = positions[thread_id, axis]
    for channel in range(g_x.shape[0]):
        local_phases[channel] = phases[channel, thread_id]

    for t in range(start, stop):
        xoroshiro128p_unit_sphere_local_float64(state, step)
        _cuda_advance_mesh(
            position,
            step,
            state,
            step_l,
            triangle_data,
            xs,
            ys,
            zs,
            subvoxel_indices,
            triangle_indices,
            iter_exc,
            thread_id,
            max_iter,
            n_sv,
            epsilon,
            perm_prob,
            n_membrane_faces,
        )
        for channel in range(g_x.shape[0]):
            local_phases[channel] += _cuda_phase_increment(
                g_x, g_y, g_z, channel, t, position[0], position[1], position[2], dt
            )

    for axis in range(3):
        positions[thread_id, axis] = position[axis]
    for channel in range(g_x.shape[0]):
        phases[channel, thread_id] = local_phases[channel]
    store_local_rng(rng_states, thread_id, state)


# Signal reduction

"""GPU reduction of walker phases to real or complex measurements."""


REDUCTION_BLOCK_SIZE = 256


@cuda.jit
def _cuda_phase_partials(
    phases, coefficients, factored, complex_signal, partial_real, partial_imag
):
    """Sum one walker tile per measurement without global atomics."""
    # Use a one-dimensional grid. CUDA gridDim.y is limited to 65,535, while
    # gridDim.x can represent the much larger flattened measurement/tile grid.
    block = cuda.blockIdx.x
    n_tiles = partial_real.shape[1]
    measurement = block // n_tiles
    tile = block - measurement * n_tiles
    lane = cuda.threadIdx.x
    walker = tile * cuda.blockDim.x + lane
    real_values = cuda.shared.array(REDUCTION_BLOCK_SIZE, np.float64)
    imag_values = cuda.shared.array(REDUCTION_BLOCK_SIZE, np.float64)

    real_value = 0.0
    imag_value = 0.0
    if walker < phases.shape[1]:
        if factored:
            phase = 0.0
            for channel in range(phases.shape[0]):
                phase += coefficients[measurement, channel] * phases[channel, walker]
        else:
            phase = phases[measurement, walker]
        real_value = math.cos(phase)
        if complex_signal:
            imag_value = math.sin(phase)

    real_values[lane] = real_value
    imag_values[lane] = imag_value
    cuda.syncthreads()
    width = cuda.blockDim.x // 2
    while width > 0:
        if lane < width:
            real_values[lane] += real_values[lane + width]
            if complex_signal:
                imag_values[lane] += imag_values[lane + width]
        cuda.syncthreads()
        width //= 2
    if lane == 0:
        partial_real[measurement, tile] = real_values[0]
        if complex_signal:
            partial_imag[measurement, tile] = imag_values[0]


@cuda.jit
def _cuda_phase_finish(partial_real, partial_imag, complex_signal, real, imag):
    """Reduce walker tiles for each measurement to one output value."""
    measurement = cuda.blockIdx.x
    lane = cuda.threadIdx.x
    real_values = cuda.shared.array(REDUCTION_BLOCK_SIZE, np.float64)
    imag_values = cuda.shared.array(REDUCTION_BLOCK_SIZE, np.float64)
    real_sum = 0.0
    imag_sum = 0.0
    for tile in range(lane, partial_real.shape[1], cuda.blockDim.x):
        real_sum += partial_real[measurement, tile]
        if complex_signal:
            imag_sum += partial_imag[measurement, tile]
    real_values[lane] = real_sum
    imag_values[lane] = imag_sum
    cuda.syncthreads()
    width = cuda.blockDim.x // 2
    while width > 0:
        if lane < width:
            real_values[lane] += real_values[lane + width]
            if complex_signal:
                imag_values[lane] += imag_values[lane + width]
        cuda.syncthreads()
        width //= 2
    if lane == 0:
        real[measurement] = real_values[0]
        if complex_signal:
            imag[measurement] = imag_values[0]


def _reduce_phases(phases, coefficients, n_measurements, complex_signal, stream):
    """Return aggregate signals after transferring only measurement totals."""
    walker_block_size = min(
        REDUCTION_BLOCK_SIZE, 1 << (phases.shape[1] - 1).bit_length()
    )
    n_tiles = (phases.shape[1] + walker_block_size - 1) // walker_block_size
    finish_block_size = min(REDUCTION_BLOCK_SIZE, 1 << (n_tiles - 1).bit_length())
    partial_shape = (n_measurements, n_tiles)
    partial_real = cuda.device_array(partial_shape, dtype=np.float64, stream=stream)
    partial_imag = (
        cuda.device_array(partial_shape, dtype=np.float64, stream=stream)
        if complex_signal
        else partial_real
    )
    d_real = cuda.device_array(n_measurements, dtype=np.float64, stream=stream)
    d_imag = (
        cuda.device_array(n_measurements, dtype=np.float64, stream=stream)
        if complex_signal
        else d_real
    )
    d_coefficients = (
        phases if coefficients is None else cuda.to_device(coefficients, stream=stream)
    )
    _cuda_phase_partials[n_tiles * n_measurements, walker_block_size, stream](
        phases,
        d_coefficients,
        coefficients is not None,
        complex_signal,
        partial_real,
        partial_imag,
    )
    _cuda_phase_finish[n_measurements, finish_block_size, stream](
        partial_real, partial_imag, complex_signal, d_real, d_imag
    )
    real = d_real.copy_to_host(stream=stream)
    if complex_signal:
        imag = d_imag.copy_to_host(stream=stream)
    stream.synchronize()
    if complex_signal:
        return real + 1j * imag
    return real


# Trajectory and mesh-cover kernels

"""Shared GPU utilities for simulation execution."""


@cuda.jit
def _cuda_record_positions(positions, records, row):
    """Save one sample-time position row without touching the walk state."""
    walker = cuda.grid(1)
    if walker >= positions.shape[0]:
        return
    for axis in range(3):
        records[row, walker, axis] = positions[walker, axis]


@cuda.jit
def _cuda_fill_mesh_cover(
    candidates,
    block_counts,
    classifier_failures,
    rng_states,
    intra,
    cell_ids,
    cell_shape,
    cell_size,
    triangle_data,
    voxel_size,
    triangle_indices,
    subvoxel_indices,
    xs,
    ys,
    zs,
    n_sv,
    n_faces,
):
    """Propose equal-volume cells, then retain only GPU-classified interiors."""
    thread_id = cuda.grid(1)
    lane = cuda.threadIdx.x
    accepted_scan = cuda.shared.array(1024, np.int32)
    point = cuda.local.array(3, np.float64)
    accepted = 0
    if thread_id < candidates.shape[0]:
        choice = xoroshiro128p_uniform_float64(rng_states, thread_id)
        cell_index = min(int(choice * cell_ids.shape[0]), cell_ids.shape[0] - 1)
        cell = cell_ids[cell_index]
        plane = cell_shape[1] * cell_shape[2]
        x = cell // plane
        remainder = cell - x * plane
        y = remainder // cell_shape[2]
        z = remainder - y * cell_shape[2]
        point[0] = (
            x + xoroshiro128p_uniform_float64(rng_states, thread_id)
        ) * cell_size[0]
        point[1] = (
            y + xoroshiro128p_uniform_float64(rng_states, thread_id)
        ) * cell_size[1]
        point[2] = (
            z + xoroshiro128p_uniform_float64(rng_states, thread_id)
        ) * cell_size[2]
        inside = _cuda_mesh_inside(
            point,
            triangle_data,
            voxel_size,
            triangle_indices,
            subvoxel_indices,
            xs,
            ys,
            zs,
            n_sv,
            n_faces,
        )
        if inside < 0:
            cuda.atomic.add(classifier_failures, 0, 1)
        elif (intra and inside == 1) or (not intra and inside == 0):
            accepted = 1

    accepted_scan[lane] = accepted
    cuda.syncthreads()
    offset = 1
    while offset < cuda.blockDim.x:
        previous = 0
        if lane >= offset:
            previous = accepted_scan[lane - offset]
        cuda.syncthreads()
        if lane >= offset:
            accepted_scan[lane] += previous
        cuda.syncthreads()
        offset *= 2
    if accepted == 1:
        candidate_index = cuda.blockIdx.x * cuda.blockDim.x + accepted_scan[lane] - 1
        for axis in range(3):
            candidates[candidate_index, axis] = point[axis]
    if lane == cuda.blockDim.x - 1:
        block_counts[cuda.blockIdx.x] = accepted_scan[lane]


# Bounded trajectory output

_TRAJECTORY_BUFFER_BYTES = 8 * 1024 * 1024


def _write_traj_row(file, positions):
    """Write one sample-time row in the documented walker-major text layout."""
    flat = positions.ravel()
    if flat.dtype == np.float64:
        # Chunk the faster Python-float formatting to bound memory use.
        for start in range(0, flat.size, 4096):
            values = flat[start : start + 4096].tolist()
            file.write(" ".join(map(str, values)) + " ")
    else:
        for value in flat:
            file.write(str(value) + " ")
    file.write("\n")


class _TrajectoryRecorder:
    """Buffer sample-time positions on the GPU and write bounded text chunks."""

    def __init__(
        self, path, n_walkers, n_steps, grid_size, block_size, stream, rotation
    ):
        row_bytes = n_walkers * 3 * np.dtype(np.float64).itemsize
        capacity = min(n_steps, max(1, _TRAJECTORY_BUFFER_BYTES // row_bytes))
        self._records = cuda.device_array(
            (capacity, n_walkers, 3), dtype=np.float64, stream=stream
        )
        self._file = open(path, "w")
        self._count = 0
        self._capacity = capacity
        self._grid_size = grid_size
        self._block_size = block_size
        self._stream = stream
        self._rotation = rotation

    def write_initial(self, positions, host_positions=None):
        if host_positions is None:
            host_positions = positions.copy_to_host(stream=self._stream)
            self._stream.synchronize()
        if self._rotation is not None:
            host_positions = host_positions @ self._rotation.T
        _write_traj_row(self._file, host_positions)

    def record(self, positions):
        _cuda_record_positions[self._grid_size, self._block_size, self._stream](
            positions, self._records, self._count
        )
        self._count += 1
        if self._count == self._capacity:
            self._flush()

    def _flush(self):
        if self._count == 0:
            return
        rows = self._records[: self._count].copy_to_host(stream=self._stream)
        self._stream.synchronize()
        for positions in rows:
            if self._rotation is not None:
                positions = positions @ self._rotation.T
            _write_traj_row(self._file, positions)
        self._count = 0

    def close(self):
        if self._file is not None:
            try:
                self._flush()
            finally:
                self._file.close()
                self._file = None


# Bounded phase-basis preparation


def _phase_error_upper_bound(gradient, basis, coefficients, dt, position_bounds, gamma):
    """Bound phase error from waveform compression for every walker.

    ``position_bounds[t]`` must bound the distance of every walker from the
    origin after diffusion interval ``t``. The interval uses the average of
    adjacent supplied gradient samples, matching the step kernels. Evaluating
    measurements in bounded chunks avoids a second full gradient allocation.
    """
    if coefficients is None:
        return 0.0
    n_measurements, n_samples, _ = gradient.shape
    if len(position_bounds) != n_samples - 1:
        raise ValueError("position_bounds must have one value per diffusion step")
    flat_basis = basis.reshape(basis.shape[0], -1)
    chunk_rows = max(1, min(32, 65536 // (n_samples * 3)))
    largest = 0.0
    for start in range(0, n_measurements, chunk_rows):
        stop = min(start + chunk_rows, n_measurements)
        residual = (coefficients[start:stop] @ flat_basis).reshape(
            stop - start, n_samples, 3
        )
        residual -= gradient[start:stop]
        interval_residual = 0.5 * (residual[:, :-1] + residual[:, 1:])
        bound_per_measurement = (
            abs(gamma)
            * dt
            * np.sum(
                np.linalg.norm(interval_residual, axis=2) * position_bounds,
                axis=1,
            )
        )
        largest = max(largest, float(np.max(bound_per_measurement)))
    return largest


def _bounded_row_basis(gradient):
    """Find a small orthonormal measurement basis without a full matrix SVD."""
    n_measurements, n_samples, _ = gradient.shape
    max_channels = min(16, (n_measurements - 1) // 2)
    matrix = gradient.reshape(n_measurements, -1).astype(np.float64, copy=False)
    row_scales = np.max(np.abs(matrix), axis=1)
    tolerance = 8 * np.finfo(gradient.dtype).eps * row_scales
    basis = np.empty((max_channels, matrix.shape[1]), dtype=np.float64)
    coefficients = np.empty((n_measurements, max_channels), dtype=np.float64)
    row_norms_squared = np.einsum("ij,ij->i", matrix, matrix)
    residual_norms_squared = row_norms_squared.copy()
    roundoff_floor = max(
        32 * np.finfo(np.float64).eps,
        8 * np.finfo(gradient.dtype).eps ** 2,
    ) * np.max(row_norms_squared)

    def reconstructed_within_tolerance(n_channels):
        # Bound temporary reconstruction storage even for long waveforms.
        chunk_rows = max(1, min(64, 65536 // matrix.shape[1]))
        for start in range(0, n_measurements, chunk_rows):
            stop = min(start + chunk_rows, n_measurements)
            difference = coefficients[start:stop, :n_channels] @ basis[:n_channels]
            difference -= matrix[start:stop]
            if np.any(np.max(np.abs(difference), axis=1) > tolerance[start:stop]):
                return False
        return True

    for channel in range(max_channels):
        pivot = int(np.argmax(residual_norms_squared))
        vector = matrix[pivot].copy()
        for earlier in range(channel):
            vector -= coefficients[pivot, earlier] * basis[earlier]
        # Reorthogonalize the new pivot against earlier basis rows.
        for earlier in range(channel):
            vector -= np.dot(vector, basis[earlier]) * basis[earlier]
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            break
        basis[channel] = vector / norm
        projection = matrix @ basis[channel]
        coefficients[:, channel] = projection
        residual_norms_squared = np.maximum(
            residual_norms_squared - projection * projection, 0.0
        )
        # Squared norm subtraction loses relative precision near a completed
        # basis; use it only as a cheap trigger for the exact row-wise check.
        small_residual = np.max(residual_norms_squared) <= roundoff_floor
        if small_residual and reconstructed_within_tolerance(channel + 1):
            reduced_basis = basis[: channel + 1]
            reduced_coefficients = coefficients[:, : channel + 1]
            return (
                np.ascontiguousarray(reduced_basis.reshape(channel + 1, n_samples, 3)),
                np.ascontiguousarray(reduced_coefficients),
            )
    return gradient, None


def _shared_waveform_basis(gradient, allow_bounded=True):
    """Return ``(basis, coefficients)`` or ``(gradient, None)``.

    A common temporal waveform with measurement-specific spatial amplitudes has
    at most three independent phase channels. A bounded row-basis fallback
    accepts other low-rank waveform families. Use either representation only
    when reconstructing the samples stays within a few input ulps and the
    phase-state reduction is substantial. ``allow_bounded`` lets the caller
    skip the more costly row-basis search for short simulations.
    """
    n_measurements, n_samples, _ = gradient.shape
    if n_measurements < 8 or gradient.dtype not in (np.float32, np.float64):
        return gradient, None

    if not np.any(gradient):
        basis = np.zeros((1, n_samples, 3), dtype=np.float64)
        coefficients = np.zeros((n_measurements, 1), dtype=np.float64)
        return basis, coefficients

    anchor_m, anchor_t, anchor_axis = np.unravel_index(
        np.argmax(np.abs(gradient)), gradient.shape
    )
    reference = gradient[anchor_m, :, anchor_axis].astype(np.float64)
    waveform = reference / reference[anchor_t]
    amplitudes = gradient[:, anchor_t, :].astype(np.float64)

    # Reject genuinely different waveforms. Rounding in float32 inputs can
    # produce a slightly larger reconstruction residual than float64 inputs.
    relative_roundoff = 8 * np.finfo(gradient.dtype).eps
    for axis in range(3):
        actual = gradient[:, :, axis]
        reconstructed = amplitudes[:, axis, None] * waveform[None, :]
        residual = np.max(np.abs(actual - reconstructed), axis=1)
        scale = np.max(np.abs(actual), axis=1)
        if np.any(residual > relative_roundoff * scale):
            return _bounded_row_basis(gradient) if allow_bounded else (gradient, None)

    active_axes = np.flatnonzero(np.any(amplitudes != 0.0, axis=0))
    n_channels = len(active_axes)
    if n_measurements < 2 * n_channels:
        return _bounded_row_basis(gradient) if allow_bounded else (gradient, None)

    basis = np.zeros((n_channels, n_samples, 3), dtype=np.float64)
    for channel, axis in enumerate(active_axes):
        basis[channel, :, axis] = waveform
    coefficients = np.ascontiguousarray(amplitudes[:, active_axes])
    return basis, coefficients
