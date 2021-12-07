"""This module contains tests of the simulations module."""

import os
import math
import numba
import pickle

import numpy as np
from numba import cuda
import numpy.testing as npt
from scipy.stats import normaltest, kstest
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
)

from .. import gradients, simulations, substrates, utils


SEED = 123


def test__cuda_dot_product():
    @cuda.jit()
    def test_kernel(a, b, dp):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        dp[thread_id] = simulations._cuda_dot_product(a[thread_id, :], b[thread_id, :])
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
        simulations._cuda_cross_product(
            a[thread_id, :], b[thread_id, :], cp[thread_id, :]
        )
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
        simulations._cuda_normalize_vector(a[thread_id, :])
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
        simulations._cuda_triangle_normal(triangle[thread_id, :], normal[thread_id, :])
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
        simulations._cuda_random_step(steps[thread_id, :], rng_states, thread_id)
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
        simulations._cuda_mat_mul(R, a[thread_id, :])
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
        d[thread_id] = simulations._cuda_line_circle_intersection(
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
        d[thread_id] = simulations._cuda_line_sphere_intersection(
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
        d[thread_id] = simulations._cuda_line_ellipsoid_intersection(
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
        ds[thread_id, :] = simulations._cuda_ray_triangle_intersection_check(
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
        simulations._cuda_reflection(
            r0[thread_id, :], step[thread_id, :], d, normal[thread_id, :], epsilon,
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
        d = simulations._cuda_ray_triangle_intersection_check(triangle, r0, step)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.float64)
            simulations._cuda_triangle_normal(triangle, normal)
            simulations._cuda_reflection(r0, step, d, normal, epsilon)
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


def test__fill_circle():
    radius = 5e-6
    N = int(1e5)
    points = simulations._fill_circle(N, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    _, p = kstest((points.ravel() + radius) / radius, "uniform")
    npt.assert_almost_equal(p, 0)
    return


def test__fill_sphere():
    radius = 5e-6
    N = int(1e5)
    points = simulations._fill_sphere(N, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    _, p = kstest((points.ravel() + radius) / radius, "uniform")
    npt.assert_almost_equal(p, 0)
    return


def test__fill_ellipsoid():
    N = int(1e5)
    a = 10e-6
    b = 2e-6
    c = 5e-6
    semiaxes = np.array([a, b, c])
    points = simulations._fill_ellipsoid(N, semiaxes)
    npt.assert_equal(np.all(np.max(points, axis=0) < [a, b, c]), True)
    npt.assert_equal(np.all(np.min(points, axis=0) > [-a, -b, -c]), True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    for i, r in enumerate([a, b, c]):
        _, p = kstest((points[:, i].ravel() + r) / r, "uniform")
        npt.assert_almost_equal(p, 0)
    return


def test__initial_positions_cylinder():
    N = int(1e3)
    r = 5e-6
    v = np.array([1.0, 0, 0])
    k = np.array([0, 1.0, 0])
    R = utils.vec2vec_rotmat(v, k)
    pos = simulations._initial_positions_cylinder(N, r, R)
    R_inv = np.linalg.inv(R)
    npt.assert_almost_equal(pos[:, 1], np.zeros(N))
    npt.assert_almost_equal(np.matmul(R_inv, pos.T)[0], np.zeros(N))
    return


def test__initial_positions_ellipsoid():
    N = int(1e3)
    r = 5e-6
    v = np.array([1.0, 0, 0])
    k = np.array([0, 1.0, 0])
    R = utils.vec2vec_rotmat(v, k)
    semiaxes = np.array([r, r, 1e-22])
    pos = simulations._initial_positions_ellipsoid(N, semiaxes, R)
    R_inv = np.linalg.inv(R)
    npt.assert_almost_equal(pos[:, 2], np.zeros(N))
    npt.assert_almost_equal(np.matmul(R_inv, pos.T)[2], np.zeros(N))
    return


def test__fill_mesh():
    n_s = int(1e4)
    mesh_path = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "sphere_mesh.pkl"
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    for n_sv in [
        np.array([1, 1, 1]),
        np.array([1, 5, 20]),
        np.array([10, 10, 10]),
    ]:
        for periodic in [True, False]:
            for padding in [np.zeros(3), np.zeros(3) + 1e-6]:

                substrate = substrates.mesh(
                    vertices, faces, periodic, padding=padding, n_sv=n_sv
                )
                points = simulations._fill_mesh(n_s, substrate, True, seed=SEED)
                r = (substrate.voxel_size - padding * 2) / 2
                points -= r + padding
                npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < r, True)
                npt.assert_almost_equal(np.mean(points, axis=0), np.zeros(3))
                points = simulations._fill_mesh(n_s, substrate, False, seed=SEED)
                points -= r + padding
                npt.assert_equal(np.min(np.linalg.norm(points, axis=1)) > 0.9 * r, True)
                npt.assert_almost_equal(np.mean(points, axis=0), np.zeros(3))
    return


def example_gradient():
    T = 80e-3
    gradient = np.zeros((1, 100, 3))
    gradient[0, 1:11, 0] = 1
    gradient[0, -11:-1, 0] = -1
    dt = T / (gradient.shape[1] - 1)
    return gradient, dt


def test_free_diffusion():

    # Signal
    n_s = int(1e5)
    n_t = int(1e3)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    bs = np.linspace(1, 2e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = substrates.free()
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals / n_s, np.exp(-bs * diffusivity), 2)

    # Walker trajectories
    n_s = int(1e4)
    n_t = int(1e2)
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "example_traj.txt"
    )
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, traj=traj_file
    )
    trajectories = np.loadtxt(traj_file)
    npt.assert_equal(trajectories.shape, (n_t + 1, n_s * 3))
    trajectories = trajectories.reshape((n_t + 1, n_s, 3))
    npt.assert_equal(np.prod(trajectories[0, :, :] == 0), 1)
    npt.assert_almost_equal(np.mean(trajectories[-1], axis=0), 0, 5)
    return


def test_cylinder_diffusion():

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "example_traj.txt"
    )
    for radius in [1e-6, 5e-6, 1e-3]:
        substrate = substrates.cylinder(
            radius=radius, orientation=np.array([1.0, 0, 0])
        )
        signals = simulations.simulation(
            n_s, diffusivity, gradient, dt, substrate, traj=traj_file
        )
        trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
        max_pos = np.max(np.linalg.norm(trajectories[..., 1::], axis=2))
        npt.assert_equal(max_pos < radius, True)
        npt.assert_almost_equal(max_pos, radius)

    # Signal compared to misst
    n_s = int(1e5)
    n_t = int(1e3)

    T = 70e-3
    gradient = np.zeros((1, 700, 3))
    gradient[0, 1:300, 0] = 1
    gradient[0, -300:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.cylinder(
        orientation=np.array([0, 0, 1.0]), radius=5e-6
    )
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)

    T = 41e-3
    gradient = np.zeros((1, 410, 3))
    gradient[0, 1:10, 0] = 1
    gradient[0, -10:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.cylinder(
        orientation=np.array([0, 0, 1.0]), radius=5e-6
    )
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_cylinder_signal_smalldelta_1ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)

    # Cylinder rotation
    n_s = int(1e5)
    n_t = int(1e3)
    gradient, dt = example_gradient()
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = substrates.cylinder(orientation=np.array([1.0, 0, 1.0]), radius=5e-6)
    signals_1 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate = substrates.cylinder(orientation=-np.array([1.0, 0, 1.0]), radius=5e-6)
    signals_2 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_1 / n_s, signals_2 / n_s)
    substrate = substrates.cylinder(orientation=-np.array([1.0, 0, 0]), radius=5e-6)
    signals_3 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_3 / n_s, np.exp(-bs * diffusivity), 2)

    return


def test_sphere_diffusion():

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "example_traj.txt"
    )
    radius = 5e-6
    substrate = substrates.sphere(radius)
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, traj=traj_file
    )
    trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
    max_pos = np.max(np.linalg.norm(trajectories, axis=2))
    npt.assert_equal(max_pos < radius, True)
    npt.assert_almost_equal(max_pos, radius)

    # Signal compared to misst
    n_s = int(1e5)
    n_t = int(1e3)

    T = 70e-3
    gradient = np.zeros((1, 700, 3))
    gradient[0, 1:300, 0] = 1
    gradient[0, -300:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.sphere(radius=5e-6)
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_sphere_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)

    T = 41e-3
    gradient = np.zeros((1, 410, 3))
    gradient[0, 1:10, 0] = 1
    gradient[0, -10:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.sphere(radius=5e-6)
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_sphere_signal_smalldelta_1ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)
    return


def test_ellipsoid_diffusion():

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "example_traj.txt"
    )
    radius = 5e-6
    semiaxes = np.ones(3) * radius
    substrate = substrates.ellipsoid(semiaxes)
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, traj=traj_file
    )
    trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
    max_pos = np.max(np.linalg.norm(trajectories, axis=2))
    npt.assert_equal(max_pos < radius, True)
    npt.assert_almost_equal(max_pos, radius)

    # Compare signal to a sphere
    substrate = substrates.sphere(radius)
    signals_sphere = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals, signals_sphere)
    return


def test_mesh_diffusion():

    n_s = int(1e4)
    n_t = int(1e3)
    diffusivity = 2e-9

    mesh_path = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "cylinder_mesh_closed.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]

    T = 70e-3
    gradient = np.zeros((1, 700, 3))
    gradient[0, 1:300, 0] = 1
    gradient[0, -300:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)

    for periodic in [True, False]:
        for padding in [np.zeros(3), np.zeros(3) + 1e-6]:
            for n_sv in [
                np.array([1, 1, 1]),
                np.array([1, 5, 20]),
                np.array([10, 10, 10]),
            ]:

                substrate = substrates.mesh(
                    vertices,
                    faces,
                    periodic,
                    padding=padding,
                    init_pos="intra",
                    n_sv=n_sv,
                )

                signals, pos = simulations.simulation(
                    n_s, diffusivity, gradient, dt, substrate, final_pos=True
                )

                # Compare to misst signal
                misst_signals_path = os.path.join(
                    os.path.dirname(gradients.__file__),
                    "tests",
                    "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
                )
                misst_signals = np.loadtxt(misst_signals_path)
                npt.assert_almost_equal(signals / n_s, misst_signals, 2)

                # Make sure no spins leaked
                r = np.max(
                    np.linalg.norm(
                        substrate.vertices[:, 0:2]
                        - (substrate.voxel_size[0:2] - padding[0:2] * 2) / 2,
                        axis=1,
                    )
                )
                l = substrate.voxel_size[2]
                npt.assert_equal(np.min(pos[:, 2]) > 0, True)
                npt.assert_equal(np.max(pos[:, 2]) < l, True)
                npt.assert_equal(
                    np.max(
                        np.linalg.norm(
                            pos[:, 0:2] - np.max(substrate.vertices, axis=0)[0:2] / 2,
                            axis=1,
                        )
                    )
                    < r,
                    True,
                )

    # Test with a periodic mesh
    mesh_path = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "cylinder_mesh_open.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    init_pos = np.zeros((n_s, 3)) + np.array([5e-6, 5e-6, 12.5e-6])

    for padding in [np.zeros(3), np.array([1e-6, 1e-6, 0])]:
        for n_sv in [
            np.array([1, 1, 1]),
            np.array([1, 5, 20]),
            np.array([10, 10, 10]),
        ]:
            substrate = substrates.mesh(
                vertices,
                faces,
                init_pos=init_pos + padding,
                periodic=True,
                padding=padding,
                n_sv=n_sv,
            )
            signals, pos = simulations.simulation(
                n_s, diffusivity, gradient, dt, substrate, final_pos=True
            )
            r = np.max(
                np.linalg.norm(
                    substrate.vertices[:, 0:2]
                    - (substrate.voxel_size[0:2] - padding[0:2] * 2) / 2,
                    axis=1,
                )
            )
            l = substrate.voxel_size[2]
            npt.assert_equal(np.min(pos[:, 2]) < 0, True)
            npt.assert_equal(np.max(pos[:, 2]) > l, True)
            npt.assert_equal(
                np.max(
                    np.linalg.norm(
                        pos[:, 0:2] - np.max(substrate.vertices, axis=0)[0:2] / 2,
                        axis=1,
                    )
                )
                < r,
                True,
            )

    # Test with neuron model and confirm that no spins leak
    n_s = int(1e3)
    n_t = int(1e2)
    gradient = np.ones((1, n_t, 3))
    for dt in [1e-5, 1e-3, 1e-1]:
        mesh_path = os.path.join(
            os.path.dirname(simulations.__file__), "tests", "neuron-model.pkl"
        )
        with open(mesh_path, "rb") as f:
            example_mesh = pickle.load(f)
        faces = example_mesh["faces"]
        vertices = example_mesh["vertices"]
        substrate = substrates.mesh(vertices, faces, init_pos="intra", periodic=True)
        signals, pos = simulations.simulation(
            n_s, diffusivity, gradient, dt, substrate, final_pos=True
        )
        npt.assert_equal(np.all(np.max(pos, axis=0) < substrate.voxel_size), True)
        npt.assert_equal(np.all(np.min(pos, axis=0) > 0), True)
    return

