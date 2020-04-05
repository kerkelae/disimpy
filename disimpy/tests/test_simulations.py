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
    gradient_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'example_gradient.txt')
    gradient = np.loadtxt(gradient_file)[np.newaxis,:,:]
    return gradient
    
def test_step_generation():
    
    @cuda.jit()
    def test_kernel(steps, rng_states):
        thread_id = cuda.grid(1)
        if thread_id >= steps.shape[0]:
            return
        step = cuda.local.array(3, numba.double)
        simulations.cuda_random_step(step, rng_states, thread_id)
        steps[thread_id, 0] = step[0]
        steps[thread_id, 1] = step[1]
        steps[thread_id, 2] = step[2]
        return

    # Define simulation size
    N = int(1e6)
    seed = 12345
    steps = np.zeros((N, 3))
    block_size = 256
    grid_size = int(math.ceil(N/block_size))
    stream = cuda.stream()
    # Run three simulations with different seeds
    # Seed 1
    rng_states = create_xoroshiro128p_states(grid_size*block_size,
                                             seed=seed,
                                             stream=stream)
    d_steps = cuda.to_device(np.ascontiguousarray(steps), stream=stream)
    test_kernel[grid_size, block_size, stream](d_steps, rng_states)
    steps_1 = d_steps.copy_to_host(stream=stream)
    stream.synchronize()    
    # Seed 2 (= seed 1)
    stream = cuda.stream()
    rng_states = create_xoroshiro128p_states(grid_size*block_size,
                                             seed=seed,
                                             stream=stream)
    d_steps = cuda.to_device(np.ascontiguousarray(steps), stream=stream)
    test_kernel[grid_size, block_size, stream](d_steps, rng_states)
    steps_2 = d_steps.copy_to_host(stream=stream)
    stream.synchronize()    
    # Seed 3 (!= seed 1)
    stream = cuda.stream()
    seed = 54321
    rng_states = create_xoroshiro128p_states(grid_size*block_size,
                                             seed=seed,
                                             stream=stream)
    d_steps = cuda.to_device(np.ascontiguousarray(steps), stream=stream)
    test_kernel[grid_size, block_size, stream](d_steps, rng_states)
    steps_3 = d_steps.copy_to_host(stream=stream)
    stream.synchronize()    
    # Assert that estimates aren't biased and that seed setting works
    npt.assert_almost_equal(np.sum(steps_1/N, axis=0), 0, 2)
    npt.assert_almost_equal(np.sum(steps_2/N, axis=0), 0, 2)
    npt.assert_almost_equal(np.sum(steps_3/N, axis=0), 0, 2)
    npt.assert_equal(steps_1, steps_2)
    return
    
def test_fill_uniformly_circle():
    radius = 5e-6
    n = int(1e5)
    points = simulations.fill_uniformly_circle(n, radius)
    npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < radius, True)
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
    return
