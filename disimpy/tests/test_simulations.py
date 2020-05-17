"""This module contains unit tests of the simulations module."""

import os
import math
import numba
import numpy as np
from numba import cuda
import numpy.testing as npt
from dipy.core.geometry import vec2vec_rotmat
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float64)

from .. import simulations, gradients, utils


def load_example_gradient():
    """Helper function for loading a gradient array."""
    gradient_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'example_gradient.txt')
    gradient = np.loadtxt(gradient_file)[np.newaxis,:,:]
    return gradient
    
    
def test_cuda_dot_product():
    
    @cuda.jit()
    def test_kernel(a, b, dp):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        dp[thread_id] = simulations.cuda_dot_product(a[thread_id, :],
                                                     b[thread_id, :])
        return
    
    a = np.array([1.2, 5, 3])[np.newaxis, :]
    b = np.array([1, 3.5, -8])[np.newaxis, :]
    dp = np.zeros(1)
    stream = cuda.stream()
    test_kernel[1, 256, stream](a, b, dp)
    stream.synchronize()    
    npt.assert_almost_equal(dp[0], np.dot(a[0], b[0]))   
    

def test_cuda_normalize_vector():

    @cuda.jit()
    def test_kernel(a):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        simulations.cuda_normalize_vector(a[thread_id, :])
        return
    
    a = np.array([1.2, -5, 3])[np.newaxis, :]
    desired_a = a / np.linalg.norm(a)
    stream = cuda.stream()
    test_kernel[1, 256, stream](a)
    stream.synchronize()    
    npt.assert_almost_equal(a, desired_a)   
    

def test_cuda_random_step():

    @cuda.jit()
    def test_kernel(steps, rng_states):
        thread_id = cuda.grid(1)
        if thread_id >= steps.shape[0]:
            return
        simulations.cuda_random_step(steps[thread_id, :], rng_states, thread_id)
        return
    
    N = int(1e5)
    seeds = [12345, 12345, 123, 1]
    steps = np.zeros((4, N, 3))
    block_size = 256
    grid_size = int(math.ceil(N/block_size))
    for i, seed in enumerate(seeds):
        stream = cuda.stream()
        rng_states = create_xoroshiro128p_states(grid_size*block_size,
                                                 seed=seed,
                                                 stream=stream)
        test_kernel[grid_size, block_size, stream](steps[i, :, :], rng_states)
        stream.synchronize()
    npt.assert_equal(steps[0], steps[1])
    npt.assert_equal(np.all(steps[0] != steps[2]), True)
    npt.assert_almost_equal(np.mean(np.sum(steps[1::]/N, axis=1)), 0, 3)
    

def test_cuda_mat_mul():
    
    @cuda.jit()
    def test_kernel(a, R):
        thread_id = cuda.grid(1)
        if thread_id >= a.shape[0]:
            return
        simulations.cuda_mat_mul(a[thread_id, :], R)
        return
    
    a = np.array([1.0, 0, 0])[np.newaxis, :]  # Original direction
    b = np.array([0.20272312, 0.06456846, 0.97710504])  # Desired direction
    R = np.array([[0.20272312, -0.06456846, -0.97710504],
                  [0.06456846, 0.99653363, -0.0524561],
                  [0.97710504, -0.0524561, 0.20618949]])  # Rotation matrix
    stream = cuda.stream()
    test_kernel[1, 256, stream](a, R)
    stream.synchronize()    
    npt.assert_almost_equal(a[0], b)  
    
    
def test_cuda_line_circle_intersection():
    
    @cuda.jit()
    def test_kernel(d, r0, step, radius):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        d[thread_id] = simulations.cuda_line_circle_intersection(
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
    
    
def test_cuda_line_sphere_intersection():
    
    @cuda.jit()
    def test_kernel(d, r0, step, radius):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        d[thread_id] = simulations.cuda_line_sphere_intersection(
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
    
    
def test_cuda_line_ellipsoid_intersection():
    
    @cuda.jit()
    def test_kernel(d, r0, step, a, b, c):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        d[thread_id] = simulations.cuda_line_ellipsoid_intersection(
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
    
def test_cuda_reflection():
    return
    
def test_fill_uniformly_circle():
    radius = 5e-6
    n = int(1e5)
    points = simulations.fill_uniformly_circle(n, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)    
    return

def test_fill_uniformly_sphere():
    radius = 5e-6
    n = int(1e5)
    points = simulations.fill_uniformly_sphere(n, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)    
    return

def test_fill_uniformly_ellipsoid():
    n = int(1e5)
    a = 10e-6
    b = 2e-6
    c = 5e-6
    points = simulations.fill_uniformly_ellipsoid(n, a, b, c)
    npt.assert_equal(np.all(np.max(points, axis=0) < [a, b, c]), True)
    npt.assert_equal(np.all(np.min(points, axis=0) > [-a, -b, -c]), True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0)    
    return

def test_free_diffusion():
    # Test signal
    n_s = int(1e5)
    n_t = int(1e3)
    n_m = int(1e2)
    diffusivity = 2e-9
    gradient = load_example_gradient()
    bs = np.linspace(0, 3e9, n_m)
    gradient = np.concatenate([gradient for i in range(n_m)], axis=0)
    diffusivity = 2e-9
    dt = 80e-3/(gradient.shape[1]-1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type' : 'free'}
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals/n_s, np.exp(-bs*diffusivity), 2)
    # Test saving trajectories in a file 
    n_s = int(1e3)
    n_t = int(1e2)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'example_traj.txt')
    signals = simulations.simulation(n_s, diffusivity, gradient, dt,
                                     substrate, trajectories=traj_file)
    trajectories = np.loadtxt(traj_file)
    npt.assert_equal(trajectories.shape, (n_t, n_s*3))
    trajectories = trajectories.reshape((n_t, n_s, 3))
    npt.assert_equal(np.prod(trajectories[0,:,:] == 0), 1)
    npt.assert_almost_equal(np.mean(np.sum(trajectories, axis=0)), 0, 3) 
    return

def test_cylinder_diffusion():
    
    # Define simulation parameters
    n_s = int(1e3) # Number of random walkers
    n_t = int(1e3) # Number of time points
    bs = np.linspace(0, 3e9, 1000) # b-values
    diffusivity = 2e-9 # In units of m^2/s

    # Define gradient array
    gradient = load_example_gradient()
    T = 80e-3 # Gradient duration
    dt = T/(gradient.shape[1]-1) # Timestep duration
    gradient = np.concatenate([gradient for i in range(len(bs))], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient[gradient > 1e-4] = 1
    gradient[gradient < -1e-4] = -1
    gradient = gradients.set_b(gradient, dt, bs)
    delta = np.sum(gradient[-1,:,:] > 0)*dt
    DELTA = np.min(np.where(gradient[-1,:,0]<0))*dt
    max_Gs = np.max(np.linalg.norm(gradient, axis=2), axis=1)
    
    # Add 6 more directions to use in tests
    phi = (1 + np.sqrt(5))/2
    directions = np.array([[0, 1, phi],
                           [0, 1, -phi],
                           [1, phi, 0],
                           [1, -phi, 0],
                           [phi, 0, 1],
                           [phi, 0, -1]])/np.linalg.norm([0,1,-phi])
    base_gradient = np.copy(gradient)
    for direction in directions:
        Rs = [vec2vec_rotmat(np.array([1,0,0]), direction) for _ in bs]
        gradient = np.concatenate((gradient, gradients.rotate_gradient(base_gradient, Rs)), axis=0)
    bvecs = np.concatenate((np.vstack([np.array([1,0,0]) for i in range(n_s)]),
                            np.vstack([directions[0] for i in range(n_s)]),
                            np.vstack([directions[1] for i in range(n_s)]),
                            np.vstack([directions[2] for i in range(n_s)]),
                            np.vstack([directions[3] for i in range(n_s)]),
                            np.vstack([directions[4] for i in range(n_s)]),
                            np.vstack([directions[5] for i in range(n_s)])), 
                           axis=0)
    max_Gs = np.concatenate(([max_Gs for i in range(7)]))
    
    # To compare these results to results acquired with Camino, define a scheme file
    # VERSION: STEJSKALTANNER
    # x_1 y_1 z_1 |G_1| DELTA_1 delta_1 TE_1
    # x_2 y_2 z_2 |G_2| DELTA_2 delta_2 TE_2
    #with open('disimpy/tests/camino/default.scheme', 'w+') as f:
    #    f.write('VERSION: STEJSKALTANNER')
    #for i, G in enumerate(max_Gs):
    #    with open('disimpy/tests/camino/default.scheme', 'a') as f:
    #        f.write('\n%s %s %s %s %s %s %s' %(bvecs[i,0], bvecs[i,1], bvecs[i,2], G, DELTA, delta, 81e-3))
    # The following commands were used in generating the results in tests/camino
    #datasynth -walkers 1000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 2E-6 -cylindersep 4.1E-6 -schemefile default.scheme > cyl_r2um.bfloat
    #datasynth -walkers 1000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 4E-6 -cylindersep 8.2E-6 -schemefile default.scheme > cyl_r4um.bfloat
    #datasynth -walkers 1000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 6E-6 -cylindersep 12.3E-6 -schemefile default.scheme > cyl_r6um.bfloat
    #datasynth -walkers 1000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 8E-6 -cylindersep 16.4E-6 -schemefile default.scheme > cyl_r8um.bfloat
    #datasynth -walkers 1000 -tmax 1000 -voxels 1 -p 0.0 -diffusivity 2E-9 -initial intra -substrate cylinder -cylinderrad 10E-6 -cylindersep 20.5E-6 -schemefile default.scheme > cyl_r10um.bfloat
    camino_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'camino')
    c_r2 = np.fromfile(os.path.join(camino_dir, 'cyl_r2um.bfloat'), dtype='>f')
    c_r4 = np.fromfile(os.path.join(camino_dir, 'cyl_r4um.bfloat'), dtype='>f')
    c_r6 = np.fromfile(os.path.join(camino_dir, 'cyl_r6um.bfloat'), dtype='>f')
    c_r8 = np.fromfile(os.path.join(camino_dir, 'cyl_r8um.bfloat'), dtype='>f')
    c_r10 = np.fromfile(os.path.join(camino_dir, 'cyl_r10um.bfloat'), dtype='>f')
    
    # Run simulations
    substrate = {'type' : 'cylinder',
             'orientation' : np.array([0, 0, 1.]),
             'radius' : 2e-6}
    s_r2 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 4e-6
    s_r4 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 6e-6
    s_r6 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 8e-6
    s_r8 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate['radius'] = 10e-6
    s_r10 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    
    # Repeat this with larger n_s later with more time
    npt.assert_almost_equal(c_r2/n_s, s_r2/n_s, 1)
    npt.assert_almost_equal(c_r4/n_s, s_r4/n_s, 1)
    npt.assert_almost_equal(c_r6/n_s, s_r6/n_s, 1)
    npt.assert_almost_equal(c_r8/n_s, s_r8/n_s, 1)
    npt.assert_almost_equal(c_r10/n_s, s_r10/n_s, 1)

def test_ellipsoid_diffusion():
    """Make sure ellipsoid and sphere diffusion gives the same result with 
       three equal semi-axes."""
    n_s = int(1e5)
    n_t = int(1e3)
    n_m = int(1e2)
    diffusivity = 2e-9
    radius = 10e-6
    gradient = load_example_gradient()
    bs = np.linspace(0, 3e9, n_m)
    gradient = np.concatenate([gradient for i in range(n_m)], axis=0)
    dt = 80e-3/(gradient.shape[1]-1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = {'type' : 'sphere',
                 'radius' : radius}
    s_sphere = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate = {'type' : 'ellipsoid',
                 'a' : radius,
                 'b' : radius,
                 'c' : radius,
                 'R' : np.eye(3)}
    s_ellipsoid = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(s_sphere/n_s, s_ellipsoid/n_s)
    R = vec2vec_rotmat(np.array([2,1.,4])/np.linalg.norm(np.array([2,1.,4])),
                       np.array([-2,0,.4])/np.linalg.norm(np.array([-2,0,.4])))
    substrate = {'type' : 'ellipsoid',
                 'a' : radius,
                 'b' : radius,
                 'c' : radius,
                 'R' : R}
    s_ellipsoid_R = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(s_ellipsoid/n_s, s_ellipsoid_R/n_s, 3)

# ADD MORE TESTS LATER 