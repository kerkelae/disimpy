"""This module contains code for executing diffusion MRI simulations."""

import os
import time
import contextlib
import math
import numpy as np
import numba
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float64,
                               xoroshiro128p_uniform_float64)

from . import utils, meshes
from .settings import EPSILON, MAX_ITER


GAMMA = 267.513e6


@cuda.jit(device=True)
def _cuda_dot_product(A, B):
    """Calculate the dot product between two 1D arrays of length 3."""
    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]


@cuda.jit(device=True)
def _cuda_cross_product(A, B, C):
    """Calculate the cross product between two 1D arrays of length 3."""
    C[0] = A[1] * B[2] - A[2] * B[1]
    C[1] = A[2] * B[0] - A[0] * B[2]
    C[2] = A[0] * B[1] - A[1] * B[0]
    return


@cuda.jit(device=True)
def _cuda_normalize_vector(v):
    """Scale 1D array of length 3 so that it has unit length."""
    length = math.sqrt(_cuda_dot_product(v, v))
    for i in range(3):
        v[i] = v[i] / length
    return


@cuda.jit(device=True)
def _cuda_random_step(step, rng_states, thread_id):
    """Generate a random step from a uniform distribution over a sphere."""
    for i in range(3):
        step[i] = xoroshiro128p_normal_float64(rng_states, thread_id)
    _cuda_normalize_vector(step)
    return


@cuda.jit(device=True)
def _cuda_mat_mul(R, v):
    """Multiply 1D array v of length 3 by a 3 x 3 matrix R."""
    rotated_v = cuda.local.array(3, numba.double)
    rotated_v[0] = R[0, 0] * v[0] + R[0, 1] * v[1] + R[0, 2] * v[2]
    rotated_v[1] = R[1, 0] * v[0] + R[1, 1] * v[1] + R[1, 2] * v[2]
    rotated_v[2] = R[2, 0] * v[0] + R[2, 1] * v[1] + R[2, 2] * v[2]
    for i in range(3):
        v[i] = rotated_v[i]
    return


@cuda.jit(device=True)
def _cuda_line_circle_intersection(r0, step, radius):
    """Calculate the distance from r0 to a circle centered at origin along step.
    r0 must be inside the circle."""
    A = step[0]**2 + step[1]**2
    B = 2 * (r0[0] * step[0] + r0[1] * step[1])
    C = r0[0]**2 + r0[1]**2 - radius**2
    d = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return d


@cuda.jit(device=True)
def _cuda_line_sphere_intersection(r0, step, radius):
    """Calculate the distance from r0 to a sphere centered at origin along step.
    r0 must be inside the sphere."""
    dp = _cuda_dot_product(step, r0)
    d = -dp + math.sqrt(dp**2 - (_cuda_dot_product(r0, r0) - radius**2))
    return d


@cuda.jit(device=True)
def _cuda_line_ellipsoid_intersection(r0, step, a, b, c):
    """Calculate the distance from r0 to an axis aligned ellipsoid centered at
    origin along step. r0 must be inside the ellipsoid."""
    A = (step[0] / a)**2 + (step[1] / b)**2 + (step[2] / c)**2
    B = 2 * (a**(-2) * step[0] * r0[0] + b**(-2) *
             step[1] * r0[1] + c**(-2) * step[2] * r0[2])
    C = (r0[0] / a)**2 + (r0[1] / b)**2 + (r0[2] / c)**2 - 1
    d = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return d


@numba.jit(nopython=True)
def _cuda_reflection(r0, step, d, normal):
    """Calculate reflection and update r0 and step accordingly."""
    intersection = cuda.local.array(3, numba.double)
    v = cuda.local.array(3, numba.double)
    for i in range(3):
        intersection[i] = r0[i] + d * step[i]
        v[i] = intersection[i] - r0[i]
    dp = _cuda_dot_product(v, normal)
    if dp < 0:  # Make sure r0 isn't 'behind' the normal
        for i in range(3):
            normal[i] *= -1
        dp = _cuda_dot_product(v, normal)
    for i in range(3):
        step[i] = ((v[i] - 2 * dp * normal[i] + intersection[i])
                   - intersection[i])
    _cuda_normalize_vector(step)
    for i in range(3):
        r0[i] = intersection[i] + EPSILON * d * step[i]
    return


@numba.jit(nopython=True)
def _fill_circle(n, radius, seed=123):
    """Sample n random points from a uniform distribution inside a circle."""
    np.random.seed(seed)
    i = 0
    filled = False
    points = np.zeros((n, 2))
    while not filled:
        p = (np.random.random(2) - .5) * 2 * radius
        if np.linalg.norm(p) < radius:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


@numba.jit(nopython=True)
def _fill_sphere(n, radius, seed=123):
    """Sample n random points from a uniform distribution inside a sphere."""
    np.random.seed(seed)
    i = 0
    filled = False
    points = np.zeros((n, 3))
    while not filled:
        p = (np.random.random(3) - .5) * 2 * radius
        if np.linalg.norm(p) < radius:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


@numba.jit(nopython=True)
def _fill_ellipsoid(n, a, b, c, seed=123):
    """Sample n random points from a uniform distribution inside an axis aligned
    ellipsoid with semi-axes a, b, and c."""
    np.random.seed(seed)
    i = 0
    filled = False
    points = np.zeros((n, 3))
    while not filled:
        p = (np.random.random(3) - .5) * 2 * np.array([a, b, c])
        if np.sum((p / np.array([a, b, c]))**2) < 1:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


def _initial_positions_cylinder(n_spins, radius, R, seed=123):
    """Calculate initial positions for spins in a cylinder whose orientation is
    defined by R which defines the rotation from cylinder frame to lab frame."""
    positions = np.zeros((n_spins, 3))
    positions[:, 1:3] = _fill_circle(n_spins, radius, seed)
    positions = np.matmul(R, positions.T).T
    return positions


def _initial_positions_ellipsoid(n_spins, a, b, c, R, seed=123):
    """Calculate initial positions for spins in an ellipsoid with semi-axes a,
    b, c whos whose orientation is defined by R which defines the rotation from
    ellipsoid frame to lab frame."""
    positions = _fill_ellipsoid(n_spins, a, b, c, seed)
    positions = np.matmul(R, positions.T).T
    return positions


@cuda.jit(device=True)
def _cuda_ray_triangle_intersection_check(A, B, C, r0, step):
    """Check if a ray defined by r0 and step intersets with a triangle defined
    by A, B, and C. The output is the distance in units of step length from r0
    to intersection if intersection found, nan otherwise. This function is based
    on the Moller-Trumbore algorithm."""
    T = cuda.local.array(3, numba.double)
    E_1 = cuda.local.array(3, numba.double)
    E_2 = cuda.local.array(3, numba.double)
    P = cuda.local.array(3, numba.double)
    Q = cuda.local.array(3, numba.double)
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
def ll_subvoxel_overlap_1d(xs, x1, x2):
    """For an interval [x1, x2], return the index of the lower limit of the
    overlapping subvoxels whose borders are defined by the elements of xs."""
    xmin = min(x1, x2)
    if xmin <= xs[0]:
        return 0
    elif xmin >= xs[-1]:
        ll = len(xs) - 1
        return ll
    else:
        ll = 0
        for i, x in enumerate(xs):
            if x > xmin:
                ll = i - 1
                return ll
        return ll


@cuda.jit(device=True)
def ul_subvoxel_overlap_1d(xs, x1, x2):
    """For an interval [xmin, xmax], return the index of the upper limit of the
    overlapping subvoxels whose borders are defined by the elements of xs."""
    xmax = max(x1, x2)
    if xmax >= xs[-1]:
        return len(xs) - 1
    elif xmax <= xs[0]:
        ul = 0
        return ul
    else:
        ul = len(xs) - 1
        for i, x in enumerate(xs):
            if not x < xmax:
                ul = i
                return ul
        return len(xs) - 1


@cuda.jit()
def _cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, t, gamma,
                    step_l, dt):
    """Kernel function for free diffusion."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.double)
    _cuda_random_step(step, rng_states, thread_id)
    for i in range(3):
        positions[thread_id, i] = positions[thread_id, i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (gamma * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_sphere(positions, g_x, g_y, g_z, phases, rng_states, t, gamma,
                      step_l, dt, radius, iter_exc):
    """Kernel function for diffusion inside a sphere."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.double)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    iter_idx = 0
    check_intersection = True
    while check_intersection and iter_idx < MAX_ITER:
        iter_idx += 1
        d = _cuda_line_sphere_intersection(r0, step, radius)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.double)
            for i in range(3):
                normal[i] = -(r0[i] + d * step[i])
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal)
            step_l -= d
        else:
            check_intersection = False
    if iter_idx >= MAX_ITER:
        iter_exc[thread_id] = True
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (gamma * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_cylinder(positions, g_x, g_y, g_z, phases, rng_states, t, gamma,
                        step_l, dt, radius, R, R_inv, iter_exc):
    """Kernel function for diffusion inside an infinite cylinder."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.double)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    _cuda_mat_mul(R, step)  # Move to cylinder frame
    _cuda_mat_mul(R, r0)  # Move to cylinder frame
    iter_idx = 0
    check_intersection = True
    while check_intersection and iter_idx < MAX_ITER:
        iter_idx += 1
        d = _cuda_line_circle_intersection(r0[1:3], step[1:3], radius)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.double)
            normal[0] = 0
            for i in range(1, 3):
                normal[i] = -(r0[i] + d * step[i])
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal)
            step_l -= d
        else:
            check_intersection = False
    if iter_idx >= MAX_ITER:
        iter_exc[thread_id] = True
    _cuda_mat_mul(R_inv, step)  # Move back to lab frame
    _cuda_mat_mul(R_inv, r0)  # Move back to lab frame
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (gamma * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_ellipsoid(positions, g_x, g_y, g_z, phases, rng_states, t,
                         gamma, step_l, dt, a, b, c, R, R_inv, iter_exc):
    """Kernel function for diffusion inside an ellipsoid."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.double)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    _cuda_mat_mul(R, step)  # Move to ellipsoid frame
    _cuda_mat_mul(R, r0)  # Move to ellipsoid frame
    iter_idx = 0
    check_intersection = True
    while check_intersection and iter_idx < MAX_ITER:
        iter_idx += 1
        d = _cuda_line_ellipsoid_intersection(r0, step, a, b, c)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.double)
            normal[0] = -(r0[0] + d * step[0]) / a**2
            normal[1] = -(r0[1] + d * step[1]) / b**2
            normal[2] = -(r0[2] + d * step[2]) / c**2
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal)
            step_l -= d
        else:
            check_intersection = False
    if iter_idx >= MAX_ITER:
        iter_exc[thread_id] = True
    _cuda_mat_mul(R_inv, step)  # Move back to lab frame
    _cuda_mat_mul(R_inv, r0)  # Move back to lab frame
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (gamma * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_mesh(positions, g_x, g_y, g_z, phases, rng_states, t, gamma,
                    step_l, dt, triangles, sv_borders, sv_mapping, tri_indices,
                    voxel_triangles, iter_exc):
    """Kernel function for diffusion restricted by a triangular mesh."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.double)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = cuda.local.array(3, numba.double)
    r0 = positions[thread_id, :]
    N = sv_borders.shape[1] - 1  # Number of subvoxels along each axis
    iter_idx = 0
    check_intersection = True
    while check_intersection and iter_idx < MAX_ITER:
        iter_idx += 1
        # Find relevant subvoxels for this step
        x_ll = ll_subvoxel_overlap_1d(
            sv_borders[0, :], r0[0], r0[0] + step[0] * step_l)
        x_ul = ul_subvoxel_overlap_1d(
            sv_borders[0, :], r0[0], r0[0] + step[0] * step_l)
        y_ll = ll_subvoxel_overlap_1d(
            sv_borders[1, :], r0[1], r0[1] + step[1] * step_l)
        y_ul = ul_subvoxel_overlap_1d(
            sv_borders[1, :], r0[1], r0[1] + step[1] * step_l)
        z_ll = ll_subvoxel_overlap_1d(
            sv_borders[2, :], r0[2], r0[2] + step[2] * step_l)
        z_ul = ul_subvoxel_overlap_1d(
            sv_borders[2, :], r0[2], r0[2] + step[2] * step_l)
        # Loop over relevant subvoxels
        min_d = math.inf
        min_idx = 0
        for x in range(x_ll, x_ul):
            for y in range(y_ll, y_ul):
                for z in range(z_ll, z_ul):
                    sv_idx = x * N**2 + y * N + z
                    # Find relevant triangles for this subvoxel
                    ll = sv_mapping[sv_idx, 0]
                    ul = sv_mapping[sv_idx, 1]
                    # Loop over relevant triangles
                    for i in range(ll, ul):
                        tri_idx = tri_indices[i] * 9
                        A = triangles[tri_idx:tri_idx + 3]
                        B = triangles[tri_idx + 3:tri_idx + 6]
                        C = triangles[tri_idx + 6:tri_idx + 9]
                        d = _cuda_ray_triangle_intersection_check(
                            A, B, C, r0, step)
                        if d > 0 and d < min_d:
                            min_d = d
                            min_idx = tri_idx
        # Check if step intersects with closest triangle
        if min_d < step_l:
            A = triangles[min_idx:min_idx + 3]
            B = triangles[min_idx + 3:min_idx + 6]
            C = triangles[min_idx + 6:min_idx + 9]
            normal = cuda.local.array(3, numba.double)
            normal[0] = ((B[1] - A[1]) * (C[2] - A[2]) -
                         (B[2] - A[2]) * (C[1] - A[1]))
            normal[1] = ((B[2] - A[2]) * (C[0] - A[0]) -
                         (B[0] - A[0]) * (C[2] - A[2]))
            normal[2] = ((B[0] - A[0]) * (C[1] - A[1]) -
                         (B[1] - A[1]) * (C[0] - A[0]))
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, min_d, normal)
            step_l -= min_d
        else:
            # Check that walker does not cross voxel boundary
            for i in range(0, 12):
                tri_idx = i * 9
                A = voxel_triangles[tri_idx:tri_idx + 3]
                B = voxel_triangles[tri_idx + 3:tri_idx + 6]
                C = voxel_triangles[tri_idx + 6:tri_idx + 9]
                d = _cuda_ray_triangle_intersection_check(A, B, C, r0, step)
                if d > 0 and d < step_l:
                    normal = cuda.local.array(3, numba.double)
                    normal[0] = ((B[1] - A[1]) * (C[2] - A[2]) -
                                 (B[2] - A[2]) * (C[1] - A[1]))
                    normal[1] = ((B[2] - A[2]) * (C[0] - A[0]) -
                                 (B[0] - A[0]) * (C[2] - A[2]))
                    normal[2] = ((B[0] - A[0]) * (C[1] - A[1]) -
                                 (B[1] - A[1]) * (C[0] - A[0]))
                    _cuda_normalize_vector(normal)
                    _cuda_reflection(r0, step, d, normal)
                    step_l -= d
                    break
                elif i == 11:
                    check_intersection = False
    if iter_idx >= MAX_ITER:
        iter_exc[thread_id] = True
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (gamma * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


def add_noise_to_data(data, sigma, seed=123):
    """Add Rician noise to data.

    Parameters
    ----------
    data : ndarray
        Array containing the data.
    sigma : float
        Standard deviation of noise in each channel.
    seed : int, optional
        Seed for pseudo random number generation.

    Returns
    -------
    noisy_data : ndarray
        Noisy data.
    """
    np.random.seed(seed)
    noisy_data = np.abs(
        data + np.random.normal(size=data.shape, scale=sigma, loc=0)
        + 1j * np.random.normal(size=data.shape, scale=sigma, loc=0))
    return noisy_data


def simulation(n_spins, diffusivity, gradient, dt, substrate, seed=123,
               trajectories=None, quiet=False, cuda_bs=128):
    """Execute a dMRI simulation.

    For a detailled tutorial, please see the documentation at
    https://disimpy.readthedocs.io/en/latest/tutorial.html

    Parameters
    ----------
    n_spins : int
        Number of random walkers in the simulation.
    diffusivity : float
        Diffusivity in SI units (m^2/s).
    gradient : ndarray
        Gradient array of shape (n of measurements, n of time points, 3). Array
        elements are floats representing the gradient magnitude at that time
        point in SI units (T/m).
    dt : float
        Duration of a time step in the gradient array in SI units (s).
    substrate : dict
        A dictionary defining the diffusion environment.
    seed : int, optional
        Seed for pseudo random number generation.
    trajectories : str, optional
        Path to file in which to save trajectories. Resulting file can be very
        large!
    quiet : bool, optional
        Define whether to print messages about simulation progression.
    cuda_bs : int, optional
        The size of the cuda thread block (1D).

    Returns
    -------
    signal : array_like
        Simulated signal(s).
    """

    # Confirm that Numba detects the GPU wihtout printing it
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        try:
            cuda.detect()
        except (cuda.cudadrv.driver.CudaSupportError or 
                cuda.cudadrv.driver.CudaAPIError):
            raise Exception(
                'Numba was unable to detect a CUDA GPU. To run the simulation,'
                + ' check that the requirements are met and CUDA installation'
                + ' path is correctly set up: '
                + 'https://numba.pydata.org/numba-doc/dev/cuda/overview.html')

    # Validate input parameters
    if (not isinstance(n_spins, int)) or (n_spins <= 0):
        raise ValueError('Incorrect value (%s) for parameter n_spins' % n_spins
                         + ' which has to be a positive integer.')
    if (not (isinstance(diffusivity, int) or isinstance(diffusivity, float)) or
            (diffusivity <= 0)):
        raise ValueError('Incorrect value (%s) for parameter' % diffusivity
                         + ' diffusivity which has to be a positive integer or'
                         + ' float.')
    if ((not isinstance(gradient, np.ndarray)) or (gradient.ndim != 3) or
            (gradient.shape[2] != 3) or (gradient.dtype != float)):
        raise ValueError('Incorrect value (%s) for parameter gradient.' % gradient
                         + ' Gradient array must be a floating point array of'
                         + ' shape (n of measurements, n of time points, 3).')
    if not (isinstance(dt, int) or isinstance(dt, float)) or (dt <= 0):
        raise ValueError('Incorrect value (%s) for parameter dt which has to' % dt
                         + ' be a positive integer or float.')
    if (not isinstance(substrate, dict)) or (not 'type' in substrate.keys()):
        raise ValueError('Incorrect value (%s) for parameter' % substrate
                         + ' substrate which has to be a dictionary with a key'
                         + ' \'type\' corresponding to one of the following'
                         + ' values: \'free\', \'cylinder\', \'sphere\','
                         + ' \'ellipsoid\', \'mesh\'.')
    if (not isinstance(seed, int)) or (seed <= 0):
        raise ValueError('Incorrect value (%s) for parameter seed which' % seed
                         + ' has to be a non-negative integer.')
    if trajectories:
        if not isinstance(trajectories, str):
            raise ValueError('Incorrect value (%s) for parameter' % trajectories
                             + ' trajectories which has to be a string.')
    if not isinstance(quiet, bool):
        raise ValueError('Incorrect value (%s) for parameter quiet' % quiet
                         + ' which has to be a boolean.')
    if (not isinstance(cuda_bs, int)) or (cuda_bs <= 0):
        raise ValueError('Incorrect value (%s) for parameter cuda_bs' % cuda_bs
                         + ' which has to be a positive integer.')

    if not quiet:
        print('Starting simulation.')
        if trajectories:
            print('The trajectories file will be up to %s GB'
                  % (gradient.shape[1] * n_spins * 3 * 25 / 1e9))

    # Set up cuda stream
    bs = cuda_bs  # Cuda block size (threads per block)
    gs = int(math.ceil(float(n_spins) / bs)) # Cuda grid size (blocks per grid)
    stream = cuda.stream()

    # Create pseudorandom number generator states
    rng_states = create_xoroshiro128p_states(gs * bs, seed=seed, stream=stream)

    # Calculate average gradient magnitude during steps
    gradient = (gradient[:, 0:-1, :] + gradient[:, 1::, :]) / 2

    # Move arrays to the GPU
    d_g_x = cuda.to_device(
        np.ascontiguousarray(gradient[:, :, 0]), stream=stream)
    d_g_y = cuda.to_device(
        np.ascontiguousarray(gradient[:, :, 1]), stream=stream)
    d_g_z = cuda.to_device(
        np.ascontiguousarray(gradient[:, :, 2]), stream=stream)
    d_iter_exc = cuda.to_device(np.zeros(n_spins).astype(bool))
    d_phases = cuda.to_device(
            np.ascontiguousarray(np.zeros((gradient.shape[0], n_spins))),
            stream=stream)

    # Calculate step length
    step_l = np.sqrt(6 * diffusivity * dt)

    if not quiet:
        print('Step length = %s' % step_l)
        print('Number of spins = %s' % n_spins)
        print('Number of steps = %s' % gradient.shape[1])

    if substrate['type'] == 'free':

        # Calculate initial positions
        positions = np.zeros((n_spins, 3))
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i) + ' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Run simulation
        for t in range(1, gradient.shape[1]):
            _cuda_step_free[gs, bs, stream](d_positions, d_g_x, d_g_y, d_g_z,
                                            d_phases, rng_states, t, GAMMA,
                                            step_l, dt)
            stream.synchronize()
            if trajectories:
                positions = d_positions.copy_to_host(stream=stream)
                with open(trajectories, 'a') as f:
                    [f.write(str(i) + ' ') for i in positions.ravel()]
                    f.write('\n')
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate['type'] == 'cylinder':

        # Validate substrate dictionary
        if ((not 'radius' in substrate.keys()) or
                (not 'orientation' in substrate.keys())):
            raise ValueError('Incorrect value (%s) for parameter' % substrate
                             + ' substrate which has to be a dictionary with'
                             + ' keys \'radius\' and \'orientation\' when'
                             + ' simulating diffusion inside an infinite'
                             + ' cylinder.')
        radius = substrate['radius']
        if (not isinstance(radius, float)) or (radius <= 0):
            raise ValueError('Incorrect value (%s) for cylinder radius' % radius
                             + ' which has to be a positive float.')
        orientation = substrate['orientation']
        if ((not isinstance(orientation, np.ndarray)) or
                (not np.any(orientation.shape == np.array([3, (1, 3), (3, 1)],
                                                          dtype=object))) or (orientation.dtype != float)):
            raise ValueError('Incorrect value (%s) for cylinder' % orientation
                             + ' orientation which has to be a float array of'
                             + ' length 3.')

        # Calculate rotation from lab frame to cylinder frame
        orientation /= np.linalg.norm(orientation)
        default_orientation = np.array([1.0, 0, 0])
        R = utils.vec2vec_rotmat(orientation, default_orientation)

        # Calculate rotation from cylinder frame to lab frame
        R_inv = np.linalg.inv(R)

        # Calculate initial positions
        positions = _initial_positions_cylinder(n_spins, radius, R_inv, seed)
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i) + ' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)
        
        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_cylinder[gs, bs, stream](d_positions, d_g_x, d_g_y,
                                                d_g_z, d_phases, rng_states, t,
                                                GAMMA, step_l, dt, radius, R,
                                                R_inv, d_iter_exc)
            stream.synchronize()
            if trajectories:
                positions = d_positions.copy_to_host(stream=stream)
                with open(trajectories, 'a') as f:
                    [f.write(str(i) + ' ') for i in positions.ravel()]
                    f.write('\n')
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate['type'] == 'sphere':

        # Validate substrate dictionary
        if not 'radius' in substrate.keys():
            raise ValueError('Incorrect value (%s) for parameter' % substrate
                             + ' substrate which has to be a dictionary with'
                             + ' a key \'radius\' when simulating diffusion'
                             + ' inside a sphere.')
        radius = substrate['radius']
        if (not isinstance(radius, float)) or (radius <= 0):
            raise ValueError('Incorrect value (%s) for sphere radius' % radius
                             + ' which has to be a positive float.')

        # Calculate initial positions
        positions = _fill_sphere(n_spins, radius, seed)
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i) + ' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_sphere[gs, bs, stream](d_positions, d_g_x, d_g_y, d_g_z,
                                              d_phases, rng_states, t, GAMMA,
                                              step_l, dt, radius, d_iter_exc)
            stream.synchronize()
            if trajectories:
                positions = d_positions.copy_to_host(stream=stream)
                with open(trajectories, 'a') as f:
                    [f.write(str(i) + ' ') for i in positions.ravel()]
                    f.write('\n')
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate['type'] == 'ellipsoid':

        # Validate substrate dictionary
        if ((not 'a' in substrate.keys()) or (not 'b' in substrate.keys())
                or (not 'c' in substrate.keys()) or
                (not 'R' in substrate.keys())):
            raise ValueError('Incorrect value (%s) for parameter' % substrate
                             + ' substrate which has to be a dictionary with'
                             + ' keys \'a\', \'b\', \'c\' and \'R\' when'
                             + ' simulating diffusion inside an ellipsoid.')
        a = substrate['a']
        if (not isinstance(a, float)) or (a <= 0):
            raise ValueError('Incorrect value (%s) for ellipsoid semiaxis a' % a
                             + ' which has to be a positive float.')
        b = substrate['b']
        if (not isinstance(b, float)) or (b <= 0):
            raise ValueError('Incorrect value (%s) for ellipsoid semiaxis b' % b
                             + ' which has to be a positive float.')
        c = substrate['c']
        if (not isinstance(c, float)) or (c <= 0):
            raise ValueError('Incorrect value (%s) for ellipsoid semiaxis c' % c
                             + ' which has to be a positive float.')
        R = substrate['R']
        if ((not isinstance(R, np.ndarray)) or (R.shape != (3, 3)) or
                (R.dtype != float)):
            raise ValueError('Incorrect value (%s) for rotation matrix R' % R
                             + ' which has to be a float array of shape (3, 3).'
                             )
       
        # Calculate rotation from ellipsoid frame to lab frame
        R_inv = R[:]

        # Calculate rotation from lab frame to cylinder frame
        R = np.linalg.inv(R_inv)

        # Calculate initial positions
        positions = _initial_positions_ellipsoid(n_spins, a, b, c, R_inv)
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i) + ' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_ellipsoid[gs, bs, stream](d_positions, d_g_x, d_g_y,
                                                 d_g_z, d_phases, rng_states, t,
                                                 GAMMA, step_l, dt, a, b, c, R,
                                                 R_inv, d_iter_exc)
            stream.synchronize()
            if trajectories:
                positions = d_positions.copy_to_host(stream=stream)
                with open(trajectories, 'a') as f:
                    [f.write(str(i) + ' ') for i in positions.ravel()]
                    f.write('\n')
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate['type'] == 'mesh':

        # Validate substrate dictionary
        if not 'mesh' in substrate.keys():
            raise ValueError('Incorrect value (%s) for parameter' % substrate
                             + ' substrate which has to be a dictionary with'
                             + ' at least a key \'mesh\', when simulating'
                             + ' diffusion inside an ellipsoid.')
        mesh = substrate['mesh']
        if (not isinstance(mesh, np.ndarray) or (mesh.shape[1] != 3) or
                (mesh.shape[2] != 3) or mesh.dtype != np.float):
            raise ValueError('Incorrect value (%s) for mesh which' % mesh
                             + ' which has to be a float array of shape (n of'
                             + ' triangles, 3, 3).')
        intra = False
        extra = False
        if (not 'intra' in substrate) and (not 'extra' in substrate):
            intra = True
            extra = True
        if 'intra' in substrate:
            intra = substrate['intra']
            if not isinstance(intra, bool):
                raise ValueError('Incorrect value (%s) for intra' % intra
                                 + ' which has to be a boolean.')
        if 'extra' in substrate:
            extra = substrate['extra']
            if not isinstance(extra, bool):
                raise ValueError('Incorrect value (%s) for extra' % extra
                                 + ' which has to be a boolean.')
        if (not intra) and (not extra):
            raise ValueError('Both intra and extra can not be False.')
        if 'N_sv' in substrate:
            N_sv = substrate['N_sv']
            if (not isinstance(N_sv, int)) or (N_sv <= 0):
                raise ValueError('Incorrect value (%s) for N_sv' % extra
                                 + ' which has to be a positive integer.')
        else:
            N_sv = 20

        # Calculate subvoxel division
        if not quiet:
            print("Calculating subvoxel division.", end="\r")
        sv_borders = meshes._mesh_space_subdivision(mesh, N=N_sv)
        tri_indices, sv_mapping = meshes._subvoxel_to_triangle_mapping(
            mesh, sv_borders)
        d_sv_borders = cuda.to_device(sv_borders, stream=stream)
        d_tri_indices = cuda.to_device(tri_indices, stream=stream)
        d_sv_mapping = cuda.to_device(sv_mapping, stream=stream)
        
        # Calculate initial positions
        if not quiet:
            print("Calculating initial positions.", end="\r")
        positions = meshes._fill_mesh(n_spins, mesh, sv_borders, tri_indices,
                                      sv_mapping, intra, extra)
        if not quiet:
            print("Finished calculating initial positions.")
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i) + ' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Calculate voxel boundaries as triangular mesh
        voxel_mesh = meshes._AABB_to_mesh(np.min(np.min(mesh, 0), 0),
                                          np.max(np.max(mesh, 0), 0))
        d_triangles = cuda.to_device(
            np.ascontiguousarray(mesh.ravel()), stream=stream)
        d_voxel_mesh = cuda.to_device(
            np.ascontiguousarray(voxel_mesh.ravel()), stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_mesh[gs, bs, stream](d_positions, d_g_x, d_g_y,
                                            d_g_z, d_phases, rng_states, t,
                                            GAMMA, step_l, dt, d_triangles,
                                            d_sv_borders, d_sv_mapping,
                                            d_tri_indices, d_voxel_mesh,
                                            d_iter_exc)
            time.sleep(1e-3)
            stream.synchronize()
            if trajectories:
                positions = d_positions.copy_to_host(stream=stream)
                with open(trajectories, 'a') as f:
                    [f.write(str(i) + ' ') for i in positions.ravel()]
                    f.write('\n')
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    else:
        raise ValueError('Incorrect value (%s) for parameter' % substrate
                         + ' substrate which has to be a dictionary with a key'
                         + ' \'type\' corresponding to one of the following'
                         + ' values: \'free\', \'cylinder\', \'sphere\','
                         + ' \'ellipsoid\', \'mesh\'.')

    # Check if intersection algorithm iteration limit was exceeded
    iter_exc = d_iter_exc.copy_to_host(stream=stream)
    if np.any(iter_exc):
        raise Exception('Maximum number of iterations was exceeded in the'
                        + ' intersection check algorithm.')

    # Calculate simulated signal
    phases = d_phases.copy_to_host(stream=stream)
    signals = np.real(np.sum(np.exp(1j * phases), axis=1))
    if not quiet:
        print('Simulation finished.')
    return signals
