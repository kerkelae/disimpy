import math
import numba
import numpy as np
from numba import cuda
from dipy.core.geometry import vec2vec_rotmat
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float64)

"""This is the module for running diffusion MRI simulations."""

MAX_ITER = 1e4
EPSILON = 1e-15

@cuda.jit(device=True)
def cuda_dot_product(a, b):
    """Calculate the dot product of two vectors of length 3."""
    dp = a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    return dp

@cuda.jit(device=True)
def cuda_normalize_vector(v):
    """Make vector have unit length."""
    length = math.sqrt(cuda_dot_product(v, v))
    v[0] = v[0]/length
    v[1] = v[1]/length
    v[2] = v[2]/length
    return 


@cuda.jit(device=True)
def cuda_random_step(step, rng_states, thread_id):
    """Generate a 3D random step."""
    step[0] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[1] = xoroshiro128p_normal_float64(rng_states, thread_id)
    step[2] = xoroshiro128p_normal_float64(rng_states, thread_id)
    cuda_normalize_vector(step)
    return

@cuda.jit()
def cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, t, n_spins,
                   gamma, step_l, dt):
    """Kernel function for free diffusion"""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.double)
    cuda_random_step(step, rng_states, thread_id)
    for i in range(3):
        positions[thread_id, i] = positions[thread_id, i] + step[i]*step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (gamma*dt*
                                 ((g_x[m, t]*positions[thread_id, 0])
                                  +(g_y[m, t]*positions[thread_id, 1])
                                  +(g_z[m, t]*positions[thread_id, 2])))    
    return

@cuda.jit(device=True)
def cuda_vector_rotation(v, R):
    """Apply 3 by 3 rotation matrix R on vector v of length 3."""
    rotated_v = cuda.local.array(3, numba.double)
    rotated_v[0] = R[0, 0]*v[0]+R[0, 1]*v[1]+R[0, 2]*v[2]
    rotated_v[1] = R[1, 0]*v[0]+R[1, 1]*v[1]+R[1, 2]*v[2]
    rotated_v[2] = R[2, 0]*v[0]+R[2, 1]*v[1]+R[2, 2]*v[2]
    for i in range(3):
        v[i] = rotated_v[i]
    return

@cuda.jit(device=True)
def cuda_cylinder_intersection_check(r0, step, radius):
    """Calculate distance from r0 to intersection."""
    a = step[0]**2+step[1]**2
    b = 2*(r0[0]*step[0]+r0[1]*step[1])
    c = r0[0]**2+r0[1]**2-radius**2
    d = (-b+math.sqrt(b**2-4*a*c))/(2*a)
    return d

@numba.jit(nopython=True)
def cuda_reflection(r0, step, d, normal):
    """Calculate reflected step"""
    intersection = cuda.local.array(3, numba.double)
    v = cuda.local.array(3, numba.double)
    for i in range(3):
        intersection[i] = r0[i] + d*step[i]
        v[i] = intersection[i] - r0[i]
    dp = cuda_dot_product(v, normal)
    if dp < 0:
        for i in range(3):
            normal[i] *= -1
        dp = cuda_dot_product(v, normal)
    for i in range(3):
        step[i] = (v[i]-2*dp*normal[i]+intersection[i]) - intersection[i]
    cuda_normalize_vector(step)
    for i in range(3):
        r0[i] = intersection[i]+EPSILON*step[i]
    return

@numba.jit(nopython=True)
def fill_uniformly_circle(n, radius, seed=123):
    """Sample n random points in a circle with radius r."""
    np.random.seed(seed)
    filled = False
    i = 0
    points = np.zeros((n, 2))
    while not filled:
        p = (np.random.random(2)-.5)*2*radius
        if np.linalg.norm(p) < radius:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points

def initial_positions_cylinder(n_spins, radius, orientation, seed=123):
    """Calculate positions for spins in a cylinder"""
    positions = np.zeros((n_spins, 3))
    positions[:, 1:3] = fill_uniformly_circle(n_spins, radius, seed=seed)
    R = vec2vec_rotmat(np.array([1, 0, 0]), orientation)
    positions = np.matmul(R, positions.T).T
    return positions

@cuda.jit()
def cuda_step_cylinder(positions, g_x, g_y, g_z, phases, rng_states, t, n_spins,
                       gamma, step_l, dt, orientation, radius, R, R_inv):
    """Kernel function for diffusion inside a cylinder"""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.double)
    cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    cuda_vector_rotation(step, R)
    cuda_vector_rotation(r0, R)
    idx = 0
    check_intersection = True
    while check_intersection and idx < MAX_ITER:
        idx += 1
        d = cuda_cylinder_intersection_check(r0[1:3], step[1:3], radius)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.double)
            normal[0] = 0
            for i in range(1,3):
                normal[i] = -(r0[i]+d*step[i])
            cuda_normalize_vector(normal)
            cuda_reflection(r0, step, d, normal)
            step_l -= d
        else:
            check_intersection = False    
    cuda_vector_rotation(step, R_inv)
    cuda_vector_rotation(r0, R_inv)   
    for i in range(3):
        positions[thread_id, i] = positions[thread_id, i] + step[i]*step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (gamma*dt*
                                 ((g_x[m, t]*positions[thread_id, 0])
                                  +(g_y[m, t]*positions[thread_id, 1])
                                  +(g_z[m, t]*positions[thread_id, 2])))
    return



def simulation(n_spins, diffusivity, gradient, dt, substrate,
               seed=123, trajectories=None):
    """Run dMRI simulation.
    
    Parameters
    ----------
    n_spins : int
        Number of random walkers in the simulation.
    diffusivity : float
        Diffusivity in SI units.
    gradient : ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in seconds.
    substrate : dict
        Diffusion environment.
    seed : int, optional
        Seed for pseudo random number generation.
    trajectories : str, optional
        Path to file in which to save trajectories. Resulting file can be very
        large!
        
    Returns
    -------
    signal : float
        Simulated signal.
    """
    GAMMA = 267.513e6
    diffusivity = diffusivity
    step_l = np.sqrt(6*diffusivity*dt)
    bs = 512 # Block size
    gs = int(math.ceil(float(n_spins) / bs)) # Grid size
    stream = cuda.stream()          
    rng_states = create_xoroshiro128p_states(gs*bs, seed=seed, stream=stream)
    d_phases = cuda.to_device(np.ascontiguousarray(np.zeros((gradient.shape[0], 
                                                             n_spins))),
                              stream=stream)
    d_g_x = cuda.to_device(np.ascontiguousarray(gradient[:, :, 0]),
                           stream=stream)
    d_g_y = cuda.to_device(np.ascontiguousarray(gradient[:, :, 1]),
                           stream=stream)
    d_g_z = cuda.to_device(np.ascontiguousarray(gradient[:, :, 2]),
                           stream=stream)
    if substrate['type'] == 'free':
        positions = np.zeros((n_spins, 3))
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i)+' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(np.ascontiguousarray(positions),
                                     stream=stream)
        for t in range(1, gradient.shape[1]):
            cuda_step_free[gs, bs, stream](d_positions, d_g_x, d_g_y, d_g_z,
                                           d_phases, rng_states, t, n_spins,
                                           GAMMA, step_l, dt)
            stream.synchronize()
            if trajectories:  # Update trajectories file
                positions = d_positions.copy_to_host(stream=stream)
                with open(trajectories, 'a') as f:
                    [f.write(str(i)+' ') for i in positions.ravel()]
                    f.write('\n')
            print(str(np.round((t/gradient.shape[1])*100, 0))+' %', end="\r")
    elif substrate['type'] == 'cylinder':
        radius = substrate['radius']
        orientation = substrate['orientation']
        R = vec2vec_rotmat(orientation, np.array([1,0,0]))
        R_inv = np.linalg.inv(R)
        positions = initial_positions_cylinder(n_spins, radius, orientation,
                                               seed)
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i)+' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(np.ascontiguousarray(positions),
                                     stream=stream)        
        for t in range(1, gradient.shape[1]):
            cuda_step_cylinder[gs, bs, stream](d_positions, d_g_x, d_g_y, 
                                               d_g_z, d_phases, rng_states, t,
                                               n_spins, GAMMA, step_l, dt,
                                               orientation, radius, R, R_inv)
            stream.synchronize()
            if trajectories:  # Update trajectories file
                positions = d_positions.copy_to_host(stream=stream)
                with open(trajectories, 'a') as f:
                    [f.write(str(i)+' ') for i in positions.ravel()]
                    f.write('\n')
            print(str(np.round((t/gradient.shape[1])*100, 0))+' %', end="\r")
        
    phases = d_phases.copy_to_host(stream=stream)
    signals = np.real(np.sum(np.exp(1j*phases), axis=1))
    return signals
