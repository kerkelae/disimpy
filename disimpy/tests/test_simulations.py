"""This module contains tests of the simulations module."""

import os
import math
import numba
import numpy as np
from numba import cuda
import numpy.testing as npt
from scipy.stats import normaltest, kstest
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float64)

from .. import simulations, gradients, utils


def load_example_gradient():
    T = 80e-3
    gradient = np.zeros((1, 100, 3))
    gradient[0, 1:11, 0] = 1
    gradient[0, -11:-1, 0] = -1
    dt = T / (gradient.shape[1] - 1)
    return gradient, dt


def load_example_mesh():
    mesh_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'example_mesh.npy')
    mesh = np.load(mesh_file)
    return mesh


def test__cuda_dot_product():

    @cuda.jit()
    def test_kernel(a, b, dp):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        dp[thread_id] = simulations._cuda_dot_product(
            a[thread_id, :], b[thread_id, :])
        return

    a = np.array([1.2, 5, 3])[np.newaxis, :]
    b = np.array([1, 3.5, -8])[np.newaxis, :]
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
            a[thread_id, :], b[thread_id, :], cp[thread_id, :])
        return

    a = np.array([1.2, 5, 3])[np.newaxis, :]
    b = np.array([1, 3.5, -8])[np.newaxis, :]
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

    a = np.array([1.2, -5, 3])[np.newaxis, :]
    desired_a = a / np.linalg.norm(a)
    stream = cuda.stream()
    test_kernel[1, 128, stream](a)
    stream.synchronize()
    npt.assert_almost_equal(a, desired_a)
    return


def test__cuda_random_step():

    @cuda.jit()
    def test_kernel(steps, rng_states):
        thread_id = cuda.grid(1)
        if thread_id >= steps.shape[0]:
            return
        simulations._cuda_random_step(
            steps[thread_id, :], rng_states, thread_id)
        return

    N = int(1e5)
    seeds = [1, 1, 12]
    steps = np.zeros((len(seeds), N, 3))
    block_size = 128
    grid_size = int(math.ceil(N / block_size))
    for i, seed in enumerate(seeds):
        stream = cuda.stream()
        rng_states = create_xoroshiro128p_states(
            grid_size * block_size, seed=seed, stream=stream)
        test_kernel[grid_size, block_size, stream](steps[i, :, :], rng_states)
        stream.synchronize()
    npt.assert_equal(steps[0], steps[1])
    npt.assert_equal(np.all(steps[0] != steps[2]), True)
    npt.assert_almost_equal(np.mean(np.sum(steps[1::], axis=1) / N), 0, 3)
    _, p = normaltest(steps[1::].ravel())
    npt.assert_almost_equal(p, 0)
    npt.assert_almost_equal(
        np.linalg.norm(
            steps, axis=2), np.ones(
            (len(seeds), N)))
    return


def test__cuda_mat_mul():

    @cuda.jit()
    def test_kernel(R, a):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        simulations._cuda_mat_mul(R, a[thread_id, :])
        return

    v = np.array([1.0, 0, 0])[np.newaxis, :]
    R = np.array([[0.20272312, -0.06456846, -0.97710504],
                  [0.06456846, 0.99653363, -0.0524561],
                  [0.97710504, -0.0524561, 0.20618949]])
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
            r0[thread_id, :], step, radius)
        return

    d = np.zeros(1)
    r0 = np.array([-.1, -.1])[np.newaxis, :]
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
            r0[thread_id, :], step, radius)
        return

    d = np.zeros(1)
    r0 = np.array([-.1, -.1, 0])[np.newaxis, :]
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
    def test_kernel(d, r0, step, a, b, c):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        d[thread_id] = simulations._cuda_line_ellipsoid_intersection(
            r0[thread_id, :], step, a, b, c)
        return

    d = np.zeros(1)
    r0 = np.array([-.1, -.1, 0])[np.newaxis, :]
    step = np.array([1.0, 1, 0])
    step /= np.linalg.norm(step)
    a, b, c = 1.0, 1.0, 1.0
    stream = cuda.stream()
    test_kernel[1, 256, stream](d, r0, step, a, b, c)
    stream.synchronize()
    npt.assert_almost_equal(d[0], 1.1414213562373097)
    return


def test__cuda_reflection():

    @cuda.jit()
    def test_kernel(r0, step, d, normal, epsilon):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        simulations._cuda_reflection(
            r0[thread_id, :], step[thread_id, :], d, normal[thread_id, :],
            epsilon)
        return

    r0 = np.array([0., 0., 0.])[np.newaxis, :]
    step = np.array([0., 0., 1.])[np.newaxis, :]
    d = .5
    normal = np.array([0., 1., 1.])[np.newaxis, :]
    normal /= np.linalg.norm(normal)
    stream = cuda.stream()
    test_kernel[1, 128, stream](r0, step, d, normal, 0.)
    stream.synchronize()
    npt.assert_almost_equal(step, np.array([[0., -1., 0.]]))
    npt.assert_almost_equal(r0, np.array([[0., 0., .5]]))

    r0 = np.array([0., 0., 0.])[np.newaxis, :]
    step = np.array([0., 0., 1.])[np.newaxis, :]
    stream = cuda.stream()
    test_kernel[1, 128, stream](r0, step, d, normal, .5)
    stream.synchronize()
    npt.assert_almost_equal(step, np.array([[0., -1., 0.]]))
    npt.assert_almost_equal(r0, np.array([[0., 0., .5]] + normal * .5))
    return


def test__fill_circle():
    radius = 5e-6
    N = int(1e5)
    points = simulations._fill_circle(N, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    _, p = kstest((points.ravel() + radius) / radius, 'uniform')
    npt.assert_almost_equal(p, 0)
    return


def test__fill_sphere():
    radius = 5e-6
    N = int(1e5)
    points = simulations._fill_sphere(N, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    _, p = kstest((points.ravel() + radius) / radius, 'uniform')
    npt.assert_almost_equal(p, 0)
    return


def test__fill_ellipsoid():
    N = int(1e5)
    a = 10e-6
    b = 2e-6
    c = 5e-6
    points = simulations._fill_ellipsoid(N, a, b, c)
    npt.assert_equal(np.all(np.max(points, axis=0) < [a, b, c]), True)
    npt.assert_equal(np.all(np.min(points, axis=0) > [-a, -b, -c]), True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    for i, r in enumerate([a, b, c]):
        _, p = kstest((points[:, i].ravel() + r) / r, 'uniform')
        npt.assert_almost_equal(p, 0)
    return


def test__cuda_ray_triangle_intersection_check():

    @cuda.jit()
    def test_kernel(ds, A, B, C, r0s, steps):
        thread_id = cuda.grid(1)
        if thread_id >= ds.shape[0]:
            return
        ds[thread_id, :] = simulations._cuda_ray_triangle_intersection_check(
            A, B, C, r0s[thread_id, :], steps[thread_id, :])
        return

    A = np.array([2.0, 0, 0])
    B = np.array([0, 2.0, 0])
    C = np.array([0.0, 0, 0])
    r0s = np.array([[0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [10, 10, 0]])
    steps = np.array([[0, 0, -1.0],
                      [0, 0, 1],
                      [0, 0, -.1],
                      [1.0, 1.0, 0],
                      [0, 0, 1.0]])
    ds = np.zeros((5, 1))
    stream = cuda.stream()
    test_kernel[1, 256, stream](ds, A, B, C, r0s, steps)
    stream.synchronize()
    npt.assert_almost_equal(ds, np.array([[1, -1, 10, np.nan, np.nan]]).T)
    return


def test__initial_positions_cylinder():
    N = int(1e3)
    r = 5e-6
    v = np.array([1., 0, 0])
    k = np.array([0, 1., 0])
    R = utils.vec2vec_rotmat(v, k)
    pos = simulations._initial_positions_cylinder(N, r, R)
    R_inv = np.linalg.inv(R)
    npt.assert_almost_equal(pos[:, 1], np.zeros(N))
    npt.assert_almost_equal(np.matmul(R_inv, pos.T)[0], np.zeros(N))
    return


def test__initial_positions_ellipsoid():
    N = int(1e3)
    r = 5e-6
    v = np.array([1., 0, 0])
    k = np.array([0, 1., 0])
    R = utils.vec2vec_rotmat(v, k)
    pos = simulations._initial_positions_ellipsoid(N, r, r, 1e-22, R)
    R_inv = np.linalg.inv(R)
    npt.assert_almost_equal(pos[:, 2], np.zeros(N))
    npt.assert_almost_equal(np.matmul(R_inv, pos.T)[2], np.zeros(N))
    return


def test__mesh_space_subdivision():
    mesh = load_example_mesh()
    for N in [10, 20, 30]:
        sv_borders = simulations._mesh_space_subdivision(mesh, N=N)
        npt.assert_equal(sv_borders.shape, (3, N + 1))
        for i in range(3):
            npt.assert_equal(
                sv_borders[i], np.linspace(
                    np.min(np.min(mesh, 0), 0)[i],
                    np.max(np.max(mesh, 0), 0)[i], N + 1))
    return


def test__interval_sv_overlap_1d():
    xs = np.arange(0, 11)
    inputs = [[0, 0], [10, 10], [2, -2], [7.2, 16]]
    outputs = [(0, 1), (9, 10), (0, 2), (7, 10)]
    for i, (x1, x2) in enumerate(inputs):
        ll, ul = simulations._interval_sv_overlap_1d(xs, x1, x2)
        npt.assert_equal((ll, ul), outputs[i])
    return


def test__subvoxel_to_triangle_mapping():
    mesh = load_example_mesh()
    sv_borders = simulations._mesh_space_subdivision(mesh, N=20)
    tri_indices, sv_mapping = simulations._subvoxel_to_triangle_mapping(
        mesh, sv_borders)
    desired_tri_indices = np.loadtxt(
        os.path.join(os.path.dirname(simulations.__file__), 'tests',
                     'desired_tri_indices.txt'))
    desired_sv_mapping = np.loadtxt(
        os.path.join(os.path.dirname(simulations.__file__), 'tests',
                     'desired_sv_mapping.txt'))
    npt.assert_equal(tri_indices, desired_tri_indices)
    npt.assert_equal(sv_mapping, desired_sv_mapping)
    return


def test__c_cross():
    np.random.seed(123)
    for _ in range(10):
        A = np.random.random(3) - .5
        B = np.random.random(3) - .5
        C = simulations._c_cross(A, B)
        npt.assert_almost_equal(C, np.cross(A, B))
    return


def test__c_dot():
    np.random.seed(123)
    for _ in range(10):
        A = np.random.random(3) - .5
        B = np.random.random(3) - .5
        C = simulations._c_dot(A, B)
        npt.assert_almost_equal(C, np.dot(A, B))
    return


def test__triangle_intersection_check():
    A = np.array([2.0, 0, 0])
    B = np.array([0, 2.0, 0])
    C = np.array([0.0, 0, 0])
    r0s = np.array([[0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [10, 10, 0]])
    steps = np.array([[0, 0, -1.0],
                      [0, 0, 1],
                      [0, 0, -.1],
                      [1.0, 1.0, 0],
                      [0, 0, 1.0]])
    desired = np.array([1, -1, 1, np.nan, np.nan])
    for i, (r0, step) in enumerate(zip(r0s, steps)):
        d = simulations._triangle_intersection_check(A, B, C, r0, step)
        npt.assert_almost_equal(d, desired[i])
    return


def test__fill_mesh():
    n_s = int(1e3)
    mesh_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'sphere_mesh.npy')
    mesh = np.load(mesh_file)
    sv_borders = simulations._mesh_space_subdivision(mesh, N=20)
    tri_indices, sv_mapping = simulations._subvoxel_to_triangle_mapping(
        mesh, sv_borders)
    points = simulations._fill_mesh(
        n_s, mesh, sv_borders, tri_indices, sv_mapping, intra=True, extra=False)
    r = np.max(mesh) / 2
    points -= r
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < r, True)
    points = simulations._fill_mesh(
        n_s, mesh, sv_borders, tri_indices, sv_mapping, intra=False, extra=True)
    r = np.max(mesh) / 2
    points -= r
    npt.assert_equal(np.min(np.linalg.norm(points, axis=1)) > .9 * r, True)
    mesh_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests',
        'cyl_mesh_r5um_l25um_closed.npy')
    mesh = np.load(mesh_file)
    r = np.max(np.max(mesh, axis=0), axis=0)[0] / 2
    l = np.max(np.max(mesh, axis=0), axis=0)[2]
    sv_borders = simulations._mesh_space_subdivision(mesh, N=20)
    tri_indices, sv_mapping = simulations._subvoxel_to_triangle_mapping(
        mesh, sv_borders)
    points = simulations._fill_mesh(
        n_s, mesh, sv_borders, tri_indices, sv_mapping, intra=True, extra=False)
    npt.assert_equal(np.min(points[:, 2]) > 0, True)
    npt.assert_equal(np.max(points[:, 2]) < l, True)
    npt.assert_equal(
        np.max(np.linalg.norm(points[:, 0:2] - r, axis=1)) < r, True)
    points = simulations._fill_mesh(
        n_s, mesh, sv_borders, tri_indices, sv_mapping, intra=False, extra=True)
    npt.assert_equal(
        np.min(np.linalg.norm(points[:, 0:2] - r, axis=1)) / .9 > r, True)
    return


def test__AABB_to_mesh():
    A = np.array([0, 0, 0])
    B = np.array([np.pi, np.pi / 2, np.pi * 2])
    mesh = simulations._AABB_to_mesh(A, B)
    desired = np.array([[[0., 0., 0., ],
                         [3.14159265, 0., 0.],
                         [3.14159265, 1.57079633, 0.]],
                        [[0., 0., 0.],
                         [0., 1.57079633, 0.],
                         [3.14159265, 1.57079633, 0.]],
                        [[0., 0., 6.28318531],
                         [3.14159265, 0., 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 0., 6.28318531],
                         [0., 1.57079633, 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 0., 0.],
                         [3.14159265, 0., 0.],
                         [3.14159265, 0., 6.28318531]],
                        [[0., 0., 0.],
                         [0., 0., 6.28318531],
                         [3.14159265, 0., 6.28318531]],
                        [[0., 1.57079633, 0.],
                         [3.14159265, 1.57079633, 0.],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 1.57079633, 0.],
                         [0., 1.57079633, 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 0., 0.],
                         [0., 1.57079633, 0.],
                         [0., 1.57079633, 6.28318531]],
                        [[0., 0., 0.],
                         [0., 0., 6.28318531],
                         [0., 1.57079633, 6.28318531]],
                        [[3.14159265, 0., 0.],
                         [3.14159265, 1.57079633, 0.],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[3.14159265, 0., 0.],
                         [3.14159265, 0., 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]]])
    npt.assert_almost_equal(mesh, desired)
    return


def test_free_diffusion():

    # Signal
    n_s = int(1e5)
    n_t = int(1e3)
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type': 'free'}
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals / n_s, np.exp(-bs * diffusivity), 2)

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    gradient, dt = load_example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'example_traj.txt')
    signals = simulations.simulation(n_s, diffusivity, gradient, dt,
                                     substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file)
    npt.assert_equal(trajectories.shape, (n_t + 1, n_s * 3))
    trajectories = trajectories.reshape((n_t + 1, n_s, 3))
    npt.assert_equal(np.prod(trajectories[0, :, :] == 0), 1)
    npt.assert_almost_equal(np.mean(np.sum(trajectories, axis=0)), 0, 3)
    return


def test_cylinder_diffusion():

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'example_traj.txt')
    radius = 5e-6
    substrate = {'type': 'cylinder',
                 'orientation': np.array([1., 0, 0]),
                 'radius': radius}
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
    max_pos = np.max(np.linalg.norm(trajectories[..., 1::], axis=2))
    npt.assert_equal(max_pos < radius, True)
    npt.assert_almost_equal(max_pos, radius)

    # Signal minimum with short pulses
    n_s = int(1e5)
    n_t = int(1e3)
    radius = 10e-6
    T = 501e-3
    gradient = np.zeros((1, n_t, 3))
    gradient[0, 1:2, 0] = 1
    gradient[0, -2:-1, 0] = -1
    dt = T / (gradient.shape[1] - 1)
    bs = np.linspace(1, 1e11, 250)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient = gradients.set_b(gradient, dt, bs)
    q = gradients.calc_q(gradient, dt)
    qs = np.max(np.linalg.norm(q, axis=2), axis=1)
    substrate = {'type': 'cylinder',
                 'orientation': np.array([0, 0, 1.0]),
                 'radius': radius}
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    minimum = 1e-6 * .61 * 2 * np.pi / radius
    npt.assert_almost_equal(qs[np.argmin(signals)] * 1e-6, minimum, 2)

    # Cylinder rotation
    n_s = int(1e5)
    n_t = int(1e3)
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type': 'cylinder',
                 'orientation': np.array([1., 0, 1.]),
                 'radius': 5e-6}
    signals_1 = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    substrate = {'type': 'cylinder',
                 'orientation': - np.array([1., 0, 1.]),
                 'radius': 5e-6}
    signals_2 = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_1 / n_s, signals_2 / n_s)
    substrate = {'type': 'cylinder',
                 'orientation': np.array([1., 0, 0]),
                 'radius': 5e-6}
    signals_3 = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_3 / n_s, np.exp(-bs * diffusivity), 2)
    return


def test_sphere_diffusion():
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'example_traj.txt')
    radius = 5e-6
    substrate = {'type': 'sphere',
                 'radius': radius}
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
    max_pos = np.max(np.linalg.norm(trajectories, axis=2))
    npt.assert_equal(max_pos < radius, True)
    npt.assert_almost_equal(max_pos, radius)
    return


def test_ellipsoid_diffusion():

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'example_traj.txt')
    radius = 5e-6
    substrate = {'type': 'ellipsoid',
                 'a': radius,
                 'b': radius,
                 'c': radius,
                 'R': np.eye(3)}
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
    max_pos = np.max(np.linalg.norm(trajectories, axis=2))
    npt.assert_equal(max_pos < radius, True)
    npt.assert_almost_equal(max_pos, radius)

    # Compare signal to a sphere
    n_s = int(1e4)
    n_t = int(1e4)
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    radius = 5e-6
    substrate = {'type': 'ellipsoid',
                 'a': radius,
                 'b': radius,
                 'c': radius,
                 'R': np.eye(3)}
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate = {'type': 'sphere',
                 'radius': radius}
    signals_sphere = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals, signals_sphere)

    # Compare signal to a cylinder (not equal but should be close)
    v = np.array([1, 0, 0])
    k = np.array([1, 1, .5])
    k /= np.linalg.norm(k)
    R = utils.vec2vec_rotmat(v, k)
    substrate = {'type': 'ellipsoid',
                 'a': 5e6,
                 'b': radius,
                 'c': radius,
                 'R': R}
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate = {'type': 'cylinder',
                 'radius': radius,
                 'orientation': k}
    signals_cyl = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals / n_s, signals_cyl / n_s, 2)
    return


def test_mesh_diffusion():

    # Confirm that mesh does not leak
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    mesh_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests',
        'cyl_mesh_r5um_l25um_closed.npy')
    mesh = np.load(mesh_file)
    mesh -= np.min(np.min(mesh, 0), 0)
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'example_traj.txt')
    radius = 5e-6
    substrate = {'type': 'mesh',
                 'mesh': mesh,
                 'intra' : True}
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
    trajectories -= np.max(np.max(mesh, 0), 0) / 2
    max_xy = np.max(np.linalg.norm(trajectories[..., 0:2], axis=2))
    npt.assert_equal(max_xy < 5e-6, True)
    npt.assert_equal(np.max(trajectories[..., 2]) < 12.5e-6, True)
    npt.assert_equal(np.min(trajectories[..., 2]) > -12.5e-6, True)

    # Test periodic boundary conditions
    mesh_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'cyl_mesh_r5um_l25um.npy')
    mesh = np.load(mesh_file)
    mesh = np.add(mesh, - np.min(np.min(mesh, 0), 0))
    traj_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests', 'example_traj.txt')
    radius = 5e-6
    init_pos = np.ones((n_s, 3)) * np.array([5e-6, 5e-6, 12.5e-6])
    substrate = {'type': 'mesh',
                 'mesh': mesh,
                 'periodic': True,
                 'initial positions': init_pos}
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file).reshape((n_t + 1, n_s, 3))
    trajectories -= np.max(np.max(mesh, 0), 0) / 2
    max_xy = np.max(np.linalg.norm(trajectories[..., 0:2], axis=2))
    npt.assert_equal(max_xy < 5e-6, True)
    npt.assert_equal(np.max(trajectories[..., 2]) > 12.5e-6, True)
    npt.assert_equal(np.min(trajectories[..., 2]) < -12.5e-6, True)

    # Test signal against analytical cylinder
    n_s = int(1e4)
    n_t = int(1e3)
    diffusivity = 2e-9
    mesh_file = os.path.join(
        os.path.dirname(simulations.__file__), 'tests',
        'cyl_mesh_r5um_l25um_closed.npy')
    mesh = np.load(mesh_file)
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    init_pos = np.zeros((n_s, 3))
    init_pos[:, 0:2] = simulations._fill_circle(n_s, 5e-6)
    init_pos += np.max(np.max(mesh, 0), 0) / 2
    substrate = {'type': 'mesh',
                 'mesh': mesh,
                 'initial positions': init_pos}
    signals_1 = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    substrate = {'type': 'cylinder',
                 'radius': 5e-6,
                 'orientation': np.array([0, 0, 1.])}
    signals_2 = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_1 / n_s, signals_2 / n_s, 2)
    substrate = {'type': 'mesh',
                 'mesh': mesh,
                 'intra' : True}
    signals_3 = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_3 / n_s, signals_2 / n_s, 2)
    npt.assert_almost_equal(signals_3 / n_s, signals_1 / n_s, 2)
    return


def test_simulation_input_validation():
    n_spins = int(1e2)
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for i in range(100)], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, 1000)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type': 'free'}
    seed = 123
    trajectories = None
    quiet = False
    cuda_bs = 128
    npt.assert_raises(ValueError, simulations.simulation, n_spins=1.1,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=-1, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=np.zeros((2, 2)), dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=0,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate={}, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=0, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=123,
                      quiet=quiet, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=12, cuda_bs=cuda_bs)
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=.2)
    substrate = {'type': 'cylinder'}
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    substrate = {'type': 'sphere'}
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    substrate = {'type': 'ellipsoid'}
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    substrate = {'type': 'mesh'}
    npt.assert_raises(ValueError, simulations.simulation, n_spins=n_spins,
                      diffusivity=diffusivity, gradient=gradient, dt=dt,
                      substrate=substrate, seed=seed, trajectories=trajectories,
                      quiet=quiet, cuda_bs=cuda_bs)
    return
