"""CUDA device-helper tests for the GPU implementation."""

import math

import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import normaltest

from .. import _cuda
from .._cuda import create_xoroshiro128p_states, cuda

SEED = 123
pytestmark = pytest.mark.gpu


def test__cuda_dot_product():
    @cuda.jit()
    def test_kernel(a, b, dp):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        dp[thread_id] = _cuda._cuda_dot_product(a[thread_id, :], b[thread_id, :])
        return

    np.random.seed(SEED)
    for _ in range(100):
        a = (np.random.random(3) - 0.5)[np.newaxis, :]
        b = (np.random.random(3) - 0.5)[np.newaxis, :]
        dp = np.zeros(1)
        stream = cuda.stream()
        test_kernel[1, 128, stream](a, b, dp)
        stream.synchronize()
        npt.assert_almost_equal(dp[0], np.dot(a[0], b[0]))
    return


def test__cuda_cross_product():
    @cuda.jit()
    def test_kernel(a, b, cp):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        _cuda._cuda_cross_product(a[thread_id, :], b[thread_id, :], cp[thread_id, :])
        return

    np.random.seed(SEED)
    for _ in range(100):
        a = (np.random.random(3) - 0.5)[np.newaxis, :]
        b = (np.random.random(3) - 0.5)[np.newaxis, :]
        cp = np.zeros(3)[np.newaxis, :]
        stream = cuda.stream()
        test_kernel[1, 128, stream](a, b, cp)
        stream.synchronize()
        npt.assert_almost_equal(cp[0], np.cross(a[0], b[0]))
    return


def test__cuda_normalize_vector():
    @cuda.jit()
    def test_kernel(a):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        _cuda._cuda_normalize_vector(a[thread_id, :])
        return

    np.random.seed(SEED)
    for _ in range(100):
        a = (np.random.random(3) - 0.5)[np.newaxis, :]
        desired_a = a / np.linalg.norm(a)
        stream = cuda.stream()
        test_kernel[1, 128, stream](a)
        stream.synchronize()
        npt.assert_almost_equal(a, desired_a)
    return


def test__cuda_triangle_normal():
    @cuda.jit()
    def test_kernel(triangle, normal):
        thread_id = cuda.grid(1)
        if thread_id >= triangle.shape[0]:
            return
        _cuda._cuda_triangle_normal(triangle[thread_id, :], normal[thread_id, :])
        return

    np.random.seed(SEED)
    for _ in range(100):
        triangle = (np.random.random((3, 3)) - 0.5)[np.newaxis, :]
        desired = np.cross(
            triangle[0, 0] - triangle[0, 1], triangle[0, 0] - triangle[0, 2]
        )
        desired /= np.linalg.norm(desired)
        desired = desired[np.newaxis, :]
        normal = np.zeros(3)[np.newaxis, :]
        stream = cuda.stream()
        test_kernel[1, 128, stream](triangle, normal)
        stream.synchronize()
        npt.assert_almost_equal(normal, desired)
    return


def test__cuda_random_step():
    @cuda.jit()
    def test_kernel(steps, rng_states):
        thread_id = cuda.grid(1)
        if thread_id >= steps.shape[0]:
            return
        _cuda._cuda_random_step(steps[thread_id, :], rng_states, thread_id)
        return

    N = int(1e5)
    seeds = [1, 1, 12]
    steps = np.zeros((len(seeds), N, 3))
    block_size = 128
    grid_size = int(math.ceil(N / block_size))
    for i, seed in enumerate(seeds):
        stream = cuda.stream()
        rng_states = create_xoroshiro128p_states(
            grid_size * block_size, seed=seed, stream=stream
        )
        test_kernel[grid_size, block_size, stream](steps[i, :, :], rng_states)
        stream.synchronize()
    npt.assert_equal(steps[0], steps[1])
    npt.assert_equal(np.all(steps[0] != steps[2]), True)
    npt.assert_almost_equal(np.mean(np.sum(steps[1::], axis=1) / N), 0, 3)
    _, p = normaltest(steps[1::].ravel())
    npt.assert_almost_equal(p, 0)
    npt.assert_almost_equal(np.linalg.norm(steps, axis=2), np.ones((len(seeds), N)))
    return


def test__cuda_mat_mul():
    @cuda.jit()
    def test_kernel(R, a):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        _cuda._cuda_mat_mul(R, a[thread_id, :])
        return

    for _ in range(100):
        v = (np.random.random(3) - 0.5)[np.newaxis, :]
        R = np.random.random((3, 3)) - 0.5
        desired_v = np.matmul(R, v[0])
        stream = cuda.stream()
        test_kernel[1, 256, stream](R, v)
        stream.synchronize()
        npt.assert_almost_equal(v[0], desired_v)
    return


def test__cuda_line_circle_intersection():
    @cuda.jit()
    def test_kernel(d, r0, step, radius):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        d[thread_id] = _cuda._cuda_line_circle_intersection(
            r0[thread_id, :], step, radius
        )
        return

    d = np.zeros(1)
    r0 = np.array([-0.1, -0.1])[np.newaxis, :]
    step = np.array([1.0, 1])
    step /= np.linalg.norm(step)
    radius = 1.0
    stream = cuda.stream()
    test_kernel[1, 256, stream](d, r0, step, radius)
    stream.synchronize()
    npt.assert_almost_equal(d[0], 1.1414213562373097)
    return


def test__cuda_line_sphere_intersection():
    @cuda.jit()
    def test_kernel(d, r0, step, radius):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        d[thread_id] = _cuda._cuda_line_sphere_intersection(
            r0[thread_id, :], step, radius
        )
        return

    d = np.zeros(1)
    r0 = np.array([-0.1, -0.1, 0])[np.newaxis, :]
    step = np.array([1.0, 1, 0])
    step /= np.linalg.norm(step)
    radius = 1.0
    stream = cuda.stream()
    test_kernel[1, 256, stream](d, r0, step, radius)
    stream.synchronize()
    npt.assert_almost_equal(d[0], 1.1414213562373097)
    return


def test__cuda_line_ellipsoid_intersection():
    @cuda.jit()
    def test_kernel(d, r0, step, semiaxes):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        d[thread_id] = _cuda._cuda_line_ellipsoid_intersection(
            r0[thread_id, :], step, semiaxes
        )
        return

    d = np.zeros(1)
    r0 = np.array([-0.1, -0.1, 0])[np.newaxis, :]
    step = np.array([1.0, 1, 0])
    step /= np.linalg.norm(step)
    semiaxes = np.array([1.0, 1.0, 1.0])
    stream = cuda.stream()
    test_kernel[1, 256, stream](d, r0, step, semiaxes)
    stream.synchronize()
    npt.assert_almost_equal(d[0], 1.1414213562373097)
    return


def test__cuda_ray_triangle_intersection_check():
    @cuda.jit()
    def test_kernel(ds, triangle, r0s, steps):
        thread_id = cuda.grid(1)
        if thread_id >= ds.shape[0]:
            return
        ds[thread_id, 0] = _cuda._cuda_ray_triangle_intersection_check(
            triangle, r0s[thread_id, :], steps[thread_id, :]
        )
        return

    triangle = np.array([[2.0, 0, 0], [0, 2.0, 0], [0.0, 0, 0]])
    r0s = np.array(
        [
            [0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [10, 10, 0],
        ]
    )
    steps = np.array(
        [[0, 0, -1.0], [0, 0, 1], [0, 0, -0.1], [1.0, 1.0, 0], [0, 0, 1.0]]
    )
    ds = np.zeros((5, 1))
    stream = cuda.stream()
    test_kernel[1, 256, stream](ds, triangle, r0s, steps)
    stream.synchronize()
    npt.assert_almost_equal(ds, np.array([[1, -1, 10, np.nan, np.nan]]).T)
    return


def test__cuda_reflection():
    @cuda.jit()
    def test_kernel(r0, step, d, normal, epsilon):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        _cuda._cuda_reflection(
            r0[thread_id, :],
            step[thread_id, :],
            d,
            normal[thread_id, :],
            epsilon,
        )
        return

    r0 = np.array([0.0, 0.0, 0.0])[np.newaxis, :]
    step = np.array([0.0, 0.0, 1.0])[np.newaxis, :]
    d = 0.5
    normal = np.array([0.0, 1.0, 1.0])[np.newaxis, :]
    normal /= np.linalg.norm(normal)
    stream = cuda.stream()
    test_kernel[1, 128, stream](r0, step, d, normal, 0.0)
    stream.synchronize()
    npt.assert_almost_equal(step, np.array([[0.0, -1.0, 0.0]]))
    npt.assert_almost_equal(r0, np.array([[0.0, 0.0, 0.5]]))

    r0 = np.array([0.0, 0.0, 0.0])[np.newaxis, :]
    step = np.array([0.0, 0.0, 1.0])[np.newaxis, :]
    stream = cuda.stream()
    test_kernel[1, 128, stream](r0, step, d, normal, 0.5)
    stream.synchronize()
    npt.assert_almost_equal(step, np.array([[0.0, -1.0, 0.0]]))
    npt.assert_almost_equal(r0, np.array([[0.0, 0.0, 0.5]] + normal * 0.5))

    @cuda.jit()
    def test_kernel(triangle, r0, step, step_l, epsilon):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        r0 = r0[thread_id]
        step = step[thread_id]
        triangle = triangle[thread_id]
        d = _cuda._cuda_ray_triangle_intersection_check(triangle, r0, step)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, np.float64)
            _cuda._cuda_triangle_normal(triangle, normal)
            _cuda._cuda_reflection(r0, step, d, normal, epsilon)
        return

    triangle = np.zeros((3, 3))
    triangle[1, 0] = 1
    triangle[2, 1] = 1
    triangle = triangle[np.newaxis, ...]
    r0 = np.array([0, 0, 0.5])[np.newaxis, ...]
    step = np.array([0, 0, -1.0])[np.newaxis, ...]
    d = 0.5
    step_l = 1.0
    epsilon = 1e-10
    stream = cuda.stream()
    test_kernel[1, 128, stream](triangle, r0, step, step_l, epsilon)
    stream.synchronize()
    npt.assert_almost_equal(step, np.array([[0.0, 0.0, 1.0]]))
    npt.assert_almost_equal(r0, np.array([[0.0, 0.0, epsilon]]))
    return


def test__cuda_crossing():

    @cuda.jit()
    def test_kernel(triangle, r0, step, step_l, epsilon):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        r0 = r0[thread_id]
        step = step[thread_id]
        triangle = triangle[thread_id]
        d = _cuda._cuda_ray_triangle_intersection_check(triangle, r0, step)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, np.float64)
            _cuda._cuda_triangle_normal(triangle, normal)
            _cuda._cuda_crossing(r0, step, d, normal, epsilon)
        return

    triangle = np.zeros((1, 3, 3))
    triangle[:, 0, 2] = 1
    triangle[:, 1, 0] = 1
    triangle[:, 1, 2] = 1
    triangle[:, 2, 1] = 1
    triangle[:, 2, 2] = 1
    r0 = np.array([0, 0, 0])[np.newaxis, ...]
    step = np.array([0, 0, 1])[np.newaxis, ...]
    step_l = 2
    epsilon = 1e-10
    stream = cuda.stream()
    test_kernel[1, 128, stream](triangle, r0, step, step_l, epsilon)
    stream.synchronize()
    npt.assert_almost_equal(r0, np.array([[0.0, 0, 1 + epsilon]]))
    return
