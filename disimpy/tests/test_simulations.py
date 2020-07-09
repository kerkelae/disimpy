"""This module contains unit tests of the simulations module."""

import os
import math
import numba
import numpy as np
from numba import cuda
import numpy.testing as npt
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float64)

from .. import simulations, gradients, utils, meshes
from ..settings import EPSILON


def load_example_gradient():
    T = 80e-3  # Duration of gradient array
    gradient_file = os.path.join(os.path.dirname(simulations.__file__),
                                 'tests', 'example_gradient.txt')
    gradient = np.loadtxt(gradient_file)[np.newaxis, :, :]
    dt = T / (gradient.shape[1] - 1)
    return gradient, dt


def test__cuda_dot_product():

    @cuda.jit()
    def test_kernel(a, b, dp):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        dp[thread_id] = simulations._cuda_dot_product(a[thread_id, :],
                                                      b[thread_id, :])
        return

    a = np.array([1.2, 5, 3])[np.newaxis, :]
    b = np.array([1, 3.5, -8])[np.newaxis, :]
    dp = np.zeros(1)
    stream = cuda.stream()
    test_kernel[1, 256, stream](a, b, dp)
    stream.synchronize()
    npt.assert_almost_equal(dp[0], np.dot(a[0], b[0]))
    return


def test__cuda_cross_product():

    @cuda.jit()
    def test_kernel(a, b, c):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        C = cuda.local.array(3, numba.double)
        simulations._cuda_cross_product(a[thread_id, :], b[thread_id, :], C)
        for i in range(3):
            c[thread_id, i] = C[i]
        return

    a = np.array([1.2, 5, 3])[np.newaxis, :]
    b = np.array([1, 3.5, -8])[np.newaxis, :]
    c = np.zeros((1, 3))
    stream = cuda.stream()
    test_kernel[1, 1, stream](a, b, c)
    stream.synchronize()
    npt.assert_almost_equal(c[0], np.cross(a[0], b[0]))
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
    test_kernel[1, 256, stream](a)
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
    seeds = [12345, 12345, 123, 1]
    steps = np.zeros((4, N, 3))
    block_size = 256
    grid_size = int(math.ceil(N / block_size))
    for i, seed in enumerate(seeds):
        stream = cuda.stream()
        rng_states = create_xoroshiro128p_states(grid_size * block_size,
                                                 seed=seed,
                                                 stream=stream)
        test_kernel[grid_size, block_size, stream](steps[i, :, :], rng_states)
        stream.synchronize()
    npt.assert_equal(steps[0], steps[1])
    npt.assert_equal(np.all(steps[0] != steps[2]), True)
    npt.assert_almost_equal(np.mean(np.sum(steps[1::] / N, axis=1)), 0, 3)
    return


def test__cuda_mat_mul():

    @cuda.jit()
    def test_kernel(R, a):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        simulations._cuda_mat_mul(R, a[thread_id, :])
        return

    a = np.array([1.0, 0, 0])[np.newaxis, :]  # Original direction
    b = np.array([0.20272312, 0.06456846, 0.97710504])  # Desired direction
    R = np.array([[0.20272312, -0.06456846, -0.97710504],
                  [0.06456846, 0.99653363, -0.0524561],
                  [0.97710504, -0.0524561, 0.20618949]])  # Rotation matrix
    stream = cuda.stream()
    test_kernel[1, 256, stream](R, a)
    stream.synchronize()
    npt.assert_almost_equal(a[0], b)
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
    def test_kernel(r0, step, d, normal):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        simulations._cuda_reflection(r0[thread_id, :], step[thread_id, :], d,
                                     normal[thread_id, :])
        return

    r0 = np.array([0.0, 0, 0])[np.newaxis, :]
    step = np.array([0, 0, 1.0])[np.newaxis, :]
    step /= np.linalg.norm(step)
    normal = np.array([0, 1.0, 1.0])[np.newaxis, :]  # To check norm mirroring
    normal /= np.linalg.norm(normal)
    d = .5
    stream = cuda.stream()
    test_kernel[1, 128, stream](r0, step, d, normal)
    stream.synchronize()
    npt.assert_almost_equal(step, np.array([[0, -1, 0]]))
    npt.assert_almost_equal(r0, np.array([[0, -EPSILON * d, 0.5]]))
    return


def test__fill_circle():
    radius = 5e-6
    n = int(1e5)
    points = simulations._fill_circle(n, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    return


def test__fill_sphere():
    radius = 5e-6
    n = int(1e5)
    points = simulations._fill_sphere(n, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
    return


def test__fill_ellipsoid():
    n = int(1e5)
    a = 10e-6
    b = 2e-6
    c = 5e-6
    points = simulations._fill_ellipsoid(n, a, b, c)
    npt.assert_equal(np.all(np.max(points, axis=0) < [a, b, c]), True)
    npt.assert_equal(np.all(np.min(points, axis=0) > [-a, -b, -c]), True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)
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


def test_free_diffusion():
    # Test signal
    n_s = int(1e5)
    n_t = int(1e3)
    n_m = int(1e2)
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 3e9, n_m)
    gradient = np.concatenate([gradient for i in range(n_m)], axis=0)
    diffusivity = 2e-9
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type': 'free'}
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals / n_s, np.exp(-bs * diffusivity), 2)
    # Test saving trajectories in a file
    n_s = int(1e2)
    n_t = int(1e2)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'example_traj.txt')
    signals = simulations.simulation(n_s, diffusivity, gradient, dt,
                                     substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file)
    npt.assert_equal(trajectories.shape, (n_t - 1, n_s * 3))
    trajectories = trajectories.reshape((n_t - 1, n_s, 3))
    npt.assert_equal(np.prod(trajectories[0, :, :] == 0), 1)
    npt.assert_almost_equal(np.mean(np.sum(trajectories, axis=0)), 0, 3)
    return


def test_cylinder_diffusion():

    n_s = int(1e4)  # Possibly increase later
    n_t = int(1e3)
    n_m = 50
    bs = np.linspace(1, 3e9, n_m)
    diffusivity = 2e-9

    # Define gradient array
    gradient, dt = load_example_gradient()
    gradient = np.concatenate([gradient for i in range(len(bs))], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient[gradient > 1e-4] = 1
    gradient[gradient < -1e-4] = -1
    gradient = gradients.set_b(gradient, dt, bs)
    delta = np.sum(gradient[-1, :, :] > 0) * dt
    DELTA = np.min(np.where(gradient[-1, :, 0] < 0)) * dt
    max_Gs = np.max(np.linalg.norm(gradient, axis=2), axis=1)

    # Add 6 more directions to use in these tests
    phi = (1 + np.sqrt(5)) / 2
    directions = np.array([[0, 1, phi],
                           [0, 1, -phi],
                           [1, phi, 0],
                           [1, -phi, 0],
                           [phi, 0, 1],
                           [phi, 0, -1]]) / np.linalg.norm([0, 1, -phi])
    base_gradient = np.copy(gradient)
    for direction in directions:
        Rs = [utils.vec2vec_rotmat(np.array([1, 0, 0]), direction) for _ in bs]
        gradient = np.concatenate(
            (gradient, gradients.rotate_gradient(
                base_gradient, Rs)), axis=0)
    bvecs = np.concatenate((np.vstack([np.array([1,
                                                 0,
                                                 0]) for i in range(n_m)]),
                            np.vstack([directions[0] for i in range(n_m)]),
                            np.vstack([directions[1] for i in range(n_m)]),
                            np.vstack([directions[2] for i in range(n_m)]),
                            np.vstack([directions[3] for i in range(n_m)]),
                            np.vstack([directions[4] for i in range(n_m)]),
                            np.vstack([directions[5] for i in range(n_m)])),
                           axis=0)
    max_Gs = np.concatenate(([max_Gs for i in range(7)]))

    # To compare the results against Camino, a scheme file was defined

    # with open('camino/default.scheme', 'w+') as f:
    #     f.write('VERSION: STEJSKALTANNER')
    # for i, G in enumerate(max_Gs):
    #     with open('camino/default.scheme', 'a') as f:
    # f.write('\n%s %s %s %s %s %s %s' %(bvecs[i,0], bvecs[i,1], bvecs[i,2],
    # G, DELTA, delta, 81e-3))

    # according to

    # VERSION: STEJSKALTANNER
    # x_1 y_1 z_1 |G_1| DELTA_1 delta_1 TE_1
    # x_2 y_2 z_2 |G_2| DELTA_2 delta_2 TE_2

    # The following commands were used in generating the results

    # datasynth -walkers 10000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 2E-6 -cylindersep 4.1E-6 -schemefile camino/default.scheme > camino/cyl_r2um.bfloat
    # datasynth -walkers 10000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 4E-6 -cylindersep 8.2E-6 -schemefile camino/default.scheme > camino/cyl_r4um.bfloat
    # datasynth -walkers 10000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 6E-6 -cylindersep 12.3E-6 -schemefile camino/default.scheme > camino/cyl_r6um.bfloat
    # datasynth -walkers 10000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 8E-6 -cylindersep 16.4E-6 -schemefile camino/default.scheme > camino/cyl_r8um.bfloat
    # datasynth -walkers 10000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9
    # -initial intra -substrate cylinder -cylinderrad 10E-6 -cylindersep
    # 20.5E-6 -schemefile camino/default.scheme > camino/cyl_r10um.bfloat

    # Load Camino results
    camino_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        'camino')
    c_r2 = np.fromfile(os.path.join(camino_dir, 'cyl_r2um.bfloat'), dtype='>f')
    c_r4 = np.fromfile(os.path.join(camino_dir, 'cyl_r4um.bfloat'), dtype='>f')
    c_r6 = np.fromfile(os.path.join(camino_dir, 'cyl_r6um.bfloat'), dtype='>f')
    c_r8 = np.fromfile(os.path.join(camino_dir, 'cyl_r8um.bfloat'), dtype='>f')
    c_r10 = np.fromfile(
        os.path.join(
            camino_dir,
            'cyl_r10um.bfloat'),
        dtype='>f')

    # Run simulations
    substrate = {'type': 'cylinder',
                 'orientation': np.array([0, 0, 1.]),
                 'radius': 2e-6}
    s_r2 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 4e-6
    s_r4 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 6e-6
    s_r6 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 8e-6
    s_r8 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 10e-6
    s_r10 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)

    # Rough comparison
    npt.assert_almost_equal(s_r2 / n_s, c_r2 / n_s, 1)
    npt.assert_almost_equal(s_r4 / n_s, c_r4 / n_s, 1)
    npt.assert_almost_equal(s_r6 / n_s, c_r6 / n_s, 1)
    npt.assert_almost_equal(s_r8 / n_s, c_r8 / n_s, 1)
    npt.assert_almost_equal(s_r10 / n_s, c_r10 / n_s, 1)
    return


def test_ellipsoid_diffusion():
    n_s = int(1e5)
    n_t = int(1e3)
    n_m = int(1e2)
    diffusivity = 2e-9
    radius = 10e-6
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 3e9, n_m)
    gradient = np.concatenate([gradient for i in range(n_m)], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type': 'sphere',
                 'radius': radius}
    s_sphere = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    substrate = {'type': 'ellipsoid',
                 'a': radius,
                 'b': radius,
                 'c': radius,
                 'R': np.eye(3)}
    s_ellipsoid = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(s_sphere / n_s, s_ellipsoid / n_s)
    R = utils.vec2vec_rotmat(np.array([2, 1., 4]) / np.linalg.norm(np.array(
        [2, 1., 4])), np.array([-2, 0, .4]) / np.linalg.norm(np.array([-2, 0, .4])))
    substrate = {'type': 'ellipsoid',
                 'a': radius,
                 'b': radius,
                 'c': radius,
                 'R': R}
    s_ellipsoid_R = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(s_ellipsoid / n_s, s_ellipsoid_R / n_s, 3)
    return


def test_mesh_diffusion():
    # Assert that mesh sphere gives similar results as analytical sphere
    n_s = int(1e4)
    n_t = int(1e3)
    n_m = int(1e2)
    radius = 5e-6
    diffusivity = 2e-9
    gradient, dt = load_example_gradient()
    bs = np.linspace(1, 10e9, n_m)
    gradient = np.concatenate([gradient for i in range(n_m)], axis=0)
    diffusivity = 2e-9
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type': 'free'}
    substrate_1 = {'type': 'sphere', 'radius': radius}
    mesh_file = os.path.join(os.path.dirname(meshes.__file__),
                             'tests', 'sphere_mesh.ply')
    mesh = meshes.load_mesh(mesh_file)
    mesh /= np.max(mesh)
    mesh *= 2 * radius
    substrate_2 = {'type': 'mesh', 'mesh': mesh, 'intra': True}
    s_1 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate_1)
    s_2 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate_2)
    npt.assert_almost_equal(s_1 / n_s, s_2 / n_s, 1)
    # Assert that no spins escape the sphere
    n_s = int(1e2)
    traj_file = os.path.join(os.path.dirname(meshes.__file__),
                             'tests', 'example_traj.txt')
    s = simulations.simulation(n_s, diffusivity, gradient, dt, substrate_2,
                               trajectories=traj_file)
    trajectories = np.loadtxt(traj_file)
    trajectories = trajectories.reshape((trajectories.shape[0],
                                         int(trajectories.shape[1] / 3),
                                         3)) - radius
    npt.assert_equal(np.max(np.linalg.norm(trajectories, axis=2)) > radius,
                     False)
    # Assert that no spins enter the sphere
    n_s = int(1e2)
    traj_file = os.path.join(os.path.dirname(meshes.__file__),
                             'tests', 'example_traj.txt')
    substrate = {'type': 'mesh', 'mesh': mesh, 'extra': True}
    s = simulations.simulation(n_s, diffusivity, gradient, dt, substrate,
                               trajectories=traj_file)
    trajectories = np.loadtxt(traj_file)
    trajectories = trajectories.reshape((trajectories.shape[0],
                                         int(trajectories.shape[1] / 3),
                                         3)) - radius
    npt.assert_equal(np.min(np.linalg.norm(trajectories, axis=2)) < .9 * radius,
                     False)
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
