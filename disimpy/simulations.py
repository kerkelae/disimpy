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
from warnings import warn

from . import utils


# This constant defines the distance by which a random walker's position is
# separated from the surface after a collision to avoid placing the walker in
# the surface.
EPSILON = 1e-12

GAMMA = 267.513e6 # Gyromagnetic ratio of the simulated spin



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
    """Multiply 1D array v of length 3 by matrix R of size 3 x 3."""
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
    if dp > 0:  # Make sure the normal vector points against the step
        for i in range(3):
            normal[i] *= -1
        dp = _cuda_dot_product(v, normal)
    for i in range(3):
        step[i] = ((v[i] - 2 * dp * normal[i] + intersection[i])
                   - intersection[i])
    _cuda_normalize_vector(step)
    for i in range(3):
        #r0[i] = intersection[i] + EPSILON * step[i]
        r0[i] = intersection[i] + EPSILON * normal[i]
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


@cuda.jit(device=True)
def _cuda_ray_triangle_intersection_check(A, B, C, r0, step):
    """Check if a ray defined by r0 and step intersects with a triangle defined
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


def _mesh_space_subdivision(mesh, N=20):
    """Divide the mesh volume into N**3 subvoxels.

    Parameters
    ----------
    mesh : numpy.ndarray
        Triangular mesh represented by an array of shape (number of triangles,
        3, 3) where the second dimension indices correspond to different
        triangle points and the third dimension representes the Cartesian
        coordinates.
    N : int
        Number of subvoxels along each Cartesian coordinate axis.

    Returns
    -------
    sv_borders : numpy.ndarray
        Array of shape (3, N + 1) representing the boundaries between the
        subvoxels along Cartesian coordinate axes.
    """
    voxel_min = np.min(np.min(mesh, 0), 0)
    voxel_max = np.max(np.max(mesh, 0), 0)
    xs = np.linspace(voxel_min[0], voxel_max[0], N + 1)
    ys = np.linspace(voxel_min[1], voxel_max[1], N + 1)
    zs = np.linspace(voxel_min[2], voxel_max[2], N + 1)
    sv_borders = np.vstack((xs, ys, zs))
    return sv_borders


@numba.jit()
def _interval_sv_overlap_1d(xs, x1, x2):
    """Return the indices of subvoxels that overlap with interval [x1, x2].

    Parameters
    ----------
    xs : numpy.ndarray
        Array of subvoxel boundaries.
    x1 : float
        Start/end point of the interval.
    x2 : float
        End/start point of the interval.

    Returns
    -------
    ll : float
        Lowest subvoxel index of the overlapping subvoxels.
    ul : float
        Highest subvoxel index of the overlapping subvoxels.
    """
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    if xmin <= xs[0]:
        ll = 0
    elif xmin >= xs[-1]:
        ll = len(xs) - 1
    else:
        ll = 0
        for i, x in enumerate(xs):
            if x > xmin:
                ll = i - 1
                break
    if xmax >= xs[-1]:
        ul = len(xs) - 1
    elif xmax <= xs[0]:
        ul = 0
    else:
        ul = len(xs) - 1
        for i, x in enumerate(xs):
            if not x < xmax:
                ul = i
                break
    if ll != ul:
        return ll, ul
    else:
        if ll != len(xs) - 1:
            return ll, ul + 1
        else:
            return ll - 1, ul


def _subvoxel_to_triangle_mapping(mesh, sv_borders):
    """Generate a mapping between subvoxels and relevant triangles.

    Parameters
    ----------
    mesh : numpy.ndarray
        Triangular mesh represented by an array of shape (number of triangles,
        3, 3) where the second dimension indices correspond to different
        triangle points and the third dimension representes the Cartesian
        coordinates.
    sv_borders : numpy.ndarray
        Array of shape (3, N + 1) representing the boundaries between the
        subvoxels along cartesian coordinate axes.

    Returns
    -------
    tri_indices : numpy.ndarray
        1D array containing the relevant triangle indices for all subvoxels.
    sv_mapping : numpy.ndarray
        2D array that allows the relevant triangle indices of a given subvoxel
        to be located in the array tri_indices. The relevant triangle indices
        for subvoxel i are tri_indices[sv_mapping[i, 0]:sv_mapping[i, 1]].

    Notes
    -----
    This implementation finds the relevant triangles by comparing the axis
    aligned bounding boxes of the triangle to the subvoxel grid. The output is
    two arrays so that it can be passed onto CUDA kernels which do not support
    3D arrays or Python data structures like lists of lists or dictionaries.
    """
    N = sv_borders.shape[1] - 1
    xs = sv_borders[0]
    ys = sv_borders[1]
    zs = sv_borders[2]
    relevant_triangles = [[] for _ in range(N**3)]
    for i, triangle in enumerate(mesh):
        xmin = np.min(triangle[:, 0])
        xmax = np.max(triangle[:, 0])
        ymin = np.min(triangle[:, 1])
        ymax = np.max(triangle[:, 1])
        zmin = np.min(triangle[:, 2])
        zmax = np.max(triangle[:, 2])
        x_ll, x_ul = _interval_sv_overlap_1d(xs, xmin, xmax)
        y_ll, y_ul = _interval_sv_overlap_1d(ys, ymin, ymax)
        z_ll, z_ul = _interval_sv_overlap_1d(zs, zmin, zmax)
        for x in range(x_ll, x_ul):
            for y in range(y_ll, y_ul):
                for z in range(z_ll, z_ul):
                    idx = x * N**2 + y * N + z  # 1D idx of this subvoxel
                    relevant_triangles[idx].append(i)
    tri_indices = []
    sv_mapping = np.zeros((len(relevant_triangles), 2))
    counter = 0
    for i, l in enumerate(relevant_triangles):
        tri_indices += l
        sv_mapping[i, 0] = counter
        sv_mapping[i, 1] = counter + len(l)
        counter += len(l)
    return np.array(tri_indices), sv_mapping.astype(int)


@numba.jit()
def _c_cross(A, B):
    """Compiled function for cross product between two 1D arrays of length 3."""
    C = np.zeros(3)
    C[0] = A[1] * B[2] - A[2] * B[1]
    C[1] = A[2] * B[0] - A[0] * B[2]
    C[2] = A[0] * B[1] - A[1] * B[0]
    return C


@numba.jit()
def _c_dot(A, B):
    """Compiled function for dot product between two 1D arrays of length 3."""
    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]


@numba.jit()
def _triangle_intersection_check(A, B, C, r0, step):
    """Return the distance from r0 to triangle ABC along ray defined by step.

    Parameters
    ----------
    A : numpy.ndarray
        1D array of length 3 defining a point of the triangle.
    B : numpy.ndarray
        1D array of length 3 defining a point of the triangle.
    C : numpy.ndarray
        1D array of length 3 defining a point of the triangle.
    r0 : numpy.ndarray
        1D array of length 3 defining the point from which the distance to the
        triangle is calculated.
    step : numpy.ndarray
        1D array of length 3 defining the direction of the ray.

    Return
    ------
    d : float
        Distance from r0 to triangle ABC along the direction defined by step.
        Returns nan in case no intersection.

    Notes
    -----
    This function is based on the Moller-Trumbore algorithm.
    """
    step = step / math.sqrt(_c_dot(step, step))
    T = r0 - A
    E_1 = B - A
    E_2 = C - A
    P = _c_cross(step, E_2)
    Q = _c_cross(T, E_1)
    det = _c_dot(P, E_1)
    if det != 0:
        t = 1 / det * _c_dot(Q, E_2)
        u = 1 / det * _c_dot(P, T)
        v = 1 / det * _c_dot(Q, step)
        if u >= 0 and u <= 1 and v >= 0 and v <= 1 and u + v <= 1:
            return t
        else:
            return np.nan
    else:
        return np.nan


def _fill_mesh(n_s, mesh, sv_borders, tri_indices, sv_mapping, intra, extra,
               seed=12345):
    """Uniformly position points inside the mesh.

    Parameters
    ----------
    n_s : int
        Number of points.
    mesh : numpy.ndarray
        Triangular mesh represented by an array of shape (number of triangles,
        3, 3) where the second dimension indices correspond to different
        triangle points and the third dimension representes the Cartesian
        coordinates.
    sv_borders : numpy.ndarray
        Array of shape (3, N + 1) representing the boundaries between the
        subvoxels along cartesian coordinate axes.
    tri_indices : numpy.ndarray
        1D array containing the relevant triangle indices for all subvoxels.
    sv_mapping : numpy.ndarray
        2D array that allows the relevant triangle indices of a given subvoxel
        to be located in the array tri_indices. The relevant triangle indices
        for subvoxel i are tri_indices[sv_mapping[i, 0]:sv_mapping[i, 1]].
    intra : bool
        Whether to place spins inside the mesh.
    extra : bool
        Whether to place spins outside the mesh.
    seed : int
        Seed for random number generation.

    Returns
    -------
    positions : numpy.ndarray
        Array of shape (n_s, 3) containing the calculated positions.
    """
    if (not intra) and (not extra):
        raise ValueError(
            'At least one of the parameters intra and extra have to be True')
    np.random.seed(seed)
    positions = np.zeros((n_s, 3))
    N = sv_borders.shape[1] - 1
    voxel_size = np.max(np.max(mesh, 0), 0)
    if intra and extra:
        for i in range(n_s):
            positions[i, :] = np.random.random(3) * voxel_size
    else:
        for i in range(n_s):
            valid = False
            while not valid:
                intersections = 0
                r0 = np.random.random(3) * voxel_size
                #step = np.array([1., 0, 0]) * voxel_size
                step = np.random.random()
                step /= np.linalg.norm(step)
                step *= voxel_size
                x_ll, x_ul = _interval_sv_overlap_1d(
                    sv_borders[0, :], r0[0], (r0 + step)[0])
                y_ll, y_ul = _interval_sv_overlap_1d(
                    sv_borders[1, :], r0[1], (r0 + step)[1])
                z_ll, z_ul = _interval_sv_overlap_1d(
                    sv_borders[2, :], r0[2], (r0 + step)[2])
                sv_indices = []
                for x in range(x_ll, x_ul):
                    for y in range(y_ll, y_ul):
                        for z in range(z_ll, z_ul):
                            sv_indices.append(x * N**2 + y * N + z)
                counted = np.zeros(mesh.shape[0]).astype(bool)
                for sv_idx in sv_indices:
                    ll = sv_mapping[sv_idx, 0]
                    ul = sv_mapping[sv_idx, 1]
                    for j in range(ll, ul):
                        tri_idx = tri_indices[j]
                        if not counted[tri_idx]:
                            counted[tri_idx] = True
                            triangle = mesh[tri_idx]
                            A = triangle[0, :]
                            B = triangle[1, :]
                            C = triangle[2, :]
                            d = _triangle_intersection_check(A, B, C, r0, step)
                            if d > 0:
                                intersections += 1
                if intra:
                    if intersections % 2 != 0:
                        valid = True
                else:
                    if intersections % 2 == 0:
                        valid = True
            positions[i, :] = r0
    return positions


def _AABB_to_mesh(A, B):
    """Return a mesh that corresponds to an axis aligned bounding box defined by
    A and B."""
    tri_1 = np.vstack((np.array([A[0], A[1], A[2]]),
                       np.array([B[0], A[1], A[2]]),
                       np.array([B[0], B[1], A[2]])))
    tri_2 = np.vstack((np.array([A[0], A[1], A[2]]),
                       np.array([A[0], B[1], A[2]]),
                       np.array([B[0], B[1], A[2]])))
    tri_3 = np.vstack((np.array([A[0], A[1], B[2]]),
                       np.array([B[0], A[1], B[2]]),
                       np.array([B[0], B[1], B[2]])))
    tri_4 = np.vstack((np.array([A[0], A[1], B[2]]),
                       np.array([A[0], B[1], B[2]]),
                       np.array([B[0], B[1], B[2]])))
    tri_5 = np.vstack((np.array([A[0], A[1], A[2]]),
                       np.array([B[0], A[1], A[2]]),
                       np.array([B[0], A[1], B[2]])))
    tri_6 = np.vstack((np.array([A[0], A[1], A[2]]),
                       np.array([A[0], A[1], B[2]]),
                       np.array([B[0], A[1], B[2]])))
    tri_7 = np.vstack((np.array([A[0], B[1], A[2]]),
                       np.array([B[0], B[1], A[2]]),
                       np.array([B[0], B[1], B[2]])))
    tri_8 = np.vstack((np.array([A[0], B[1], A[2]]),
                       np.array([A[0], B[1], B[2]]),
                       np.array([B[0], B[1], B[2]])))
    tri_9 = np.vstack((np.array([A[0], A[1], A[2]]),
                       np.array([A[0], B[1], A[2]]),
                       np.array([A[0], B[1], B[2]])))
    tri_10 = np.vstack((np.array([A[0], A[1], A[2]]),
                        np.array([A[0], A[1], B[2]]),
                        np.array([A[0], B[1], B[2]])))
    tri_11 = np.vstack((np.array([B[0], A[1], A[2]]),
                        np.array([B[0], B[1], A[2]]),
                        np.array([B[0], B[1], B[2]])))
    tri_12 = np.vstack((np.array([B[0], A[1], A[2]]),
                        np.array([B[0], A[1], B[2]]),
                        np.array([B[0], B[1], B[2]])))
    mesh = np.concatenate((tri_1[np.newaxis, :, :],
                           tri_2[np.newaxis, :, :],
                           tri_3[np.newaxis, :, :],
                           tri_4[np.newaxis, :, :],
                           tri_5[np.newaxis, :, :],
                           tri_6[np.newaxis, :, :],
                           tri_7[np.newaxis, :, :],
                           tri_8[np.newaxis, :, :],
                           tri_9[np.newaxis, :, :],
                           tri_10[np.newaxis, :, :],
                           tri_11[np.newaxis, :, :],
                           tri_12[np.newaxis, :, :]), axis=0)
    return mesh


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
    """For an interval [x1, x2], return the index of the upper limit of the
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


@cuda.jit(device=True)
def ll_subvoxel_overlap_1d_periodic(xs, x1, x2):
    """For an interval [x1, x2], return the index of the lower limit of the
    overlapping subvoxels whose borders are defined by the elements of xs. The
    subvoxel division continues outside the voxel."""
    xmin = min(x1, x2)
    voxel_size = abs(xs[-1] - xs[0])
    n = math.floor(xmin / voxel_size)  # How many voxel widths to shift
    xmin_shifted = xmin - n * voxel_size
    ll_shifted = ll_subvoxel_overlap_1d(xs, xmin_shifted, xmin_shifted)
    ll = ll_shifted + n * (len(xs) - 1)
    return ll


@cuda.jit(device=True)
def ul_subvoxel_overlap_1d_periodic(xs, x1, x2):
    """For an interval [x1, x2], return the index of the upper limit of the
    overlapping subvoxels whose borders are defined by the elements of xs. The
    subvoxel division continues outside the voxel."""
    xmax = max(x1, x2)
    voxel_size = abs(xs[-1] - xs[0])
    n = math.floor(xmax / voxel_size)  # How many voxel widths to shift
    xmax_shifted = xmax - n * voxel_size
    ul_shifted = ul_subvoxel_overlap_1d(xs, xmax_shifted, xmax_shifted)
    ul = ul_shifted + n * (len(xs) - 1)
    return ul


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
                    voxel_triangles, iter_exc, periodic):
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
    if periodic:  # Periodic boundary conditions
        shifts = cuda.local.array(3, numba.double)
        temp_shifts = cuda.local.array(3, numba.double)
        temp_r0 = cuda.local.array(3, numba.double)
        lls = cuda.local.array(3, numba.double)
        uls = cuda.local.array(3, numba.double)
        while check_intersection and iter_idx < MAX_ITER:
            iter_idx += 1
            min_d = math.inf
            min_idx = 0
            for i in range(3):  # Find relevant subvoxels
                lls[i] = ll_subvoxel_overlap_1d_periodic(
                    sv_borders[i, :], r0[i], r0[i] + step[i] * step_l)
                uls[i] = ul_subvoxel_overlap_1d_periodic(
                    sv_borders[i, :], r0[i], r0[i] + step[i] * step_l)
            for x in range(lls[0], uls[0]):  # Loop over relevant subvoxels
                if x < 0 or x > N - 1:
                    shift_n = math.floor(x / N)
                    x -= shift_n * N
                    temp_shifts[0] = shift_n * sv_borders[0, -1]
                else:
                    temp_shifts[0] = 0
                for y in range(lls[1], uls[1]):
                    if y < 0 or y > N - 1:
                        shift_n = math.floor(y / N)
                        y -= shift_n * N
                        temp_shifts[1] = shift_n * sv_borders[1, -1]
                    else:
                        temp_shifts[1] = 0
                    for z in range(lls[2], uls[2]):
                        if z < 0 or z > N - 1:
                            shift_n = math.floor(z / N)
                            z -= shift_n * N
                            temp_shifts[2] = shift_n * sv_borders[2, -1]
                        else:
                            temp_shifts[2] = 0
                        sv_idx = int(x * N**2 + y * N + z)
                        for i in range(3):  # Shift walker
                            temp_r0[i] = r0[i] - temp_shifts[i]
                        # Loop over relevant triangles
                        for i in range(sv_mapping[sv_idx, 0],
                                       sv_mapping[sv_idx, 1]):
                            tri_idx = tri_indices[i] * 9
                            A = triangles[tri_idx:tri_idx + 3]
                            B = triangles[tri_idx + 3:tri_idx + 6]
                            C = triangles[tri_idx + 6:tri_idx + 9]
                            d = _cuda_ray_triangle_intersection_check(
                                A, B, C, temp_r0, step)
                            if d > 0 and d < min_d:
                                min_d = d
                                min_idx = tri_idx
                                for j in range(3):
                                    shifts[j] = temp_shifts[j]
            if min_d < step_l:  # Step intersects with closest triangle
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
                for i in range(3):  # Shift walker to voxel
                    r0[i] -= shifts[i]
                _cuda_reflection(r0, step, min_d, normal)
                for i in range(3):  # Shift walker back
                    r0[i] += shifts[i]
                step_l -= min_d
            else:
                check_intersection = False

    else:  # Reflective boundary conditions
        lls = cuda.local.array(3, numba.double)
        uls = cuda.local.array(3, numba.double)
        while check_intersection and iter_idx < MAX_ITER:
            iter_idx += 1
            min_d = math.inf
            min_idx = 0
            for i in range(3):  # Find relevant subvoxels
                lls[i] = ll_subvoxel_overlap_1d(
                    sv_borders[i, :], r0[i], r0[i] + step[i] * step_l)
                uls[i] = ul_subvoxel_overlap_1d(
                    sv_borders[i, :], r0[i], r0[i] + step[i] * step_l)
            for x in range(lls[0], uls[0]):  # Loop over relevant subvoxels
                for y in range(lls[1], uls[1]):
                    for z in range(lls[2], uls[2]):
                        sv_idx = x * N**2 + y * N + z
                        # Loop over relevant triangles
                        for i in range(sv_mapping[sv_idx, 0],
                                       sv_mapping[sv_idx, 1]):
                            tri_idx = tri_indices[i] * 9
                            A = triangles[tri_idx:tri_idx + 3]
                            B = triangles[tri_idx + 3:tri_idx + 6]
                            C = triangles[tri_idx + 6:tri_idx + 9]
                            d = _cuda_ray_triangle_intersection_check(
                                A, B, C, r0, step)
                            if d > 0 and d < min_d:
                                min_d = d
                                min_idx = tri_idx
            if min_d < step_l:  # Step intersects with closest triangle
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
                for i in range(0, 12):  # Check that walker doesn't leave voxel
                    tri_idx = i * 9
                    A = voxel_triangles[tri_idx:tri_idx + 3]
                    B = voxel_triangles[tri_idx + 3:tri_idx + 6]
                    C = voxel_triangles[tri_idx + 6:tri_idx + 9]
                    d = _cuda_ray_triangle_intersection_check(
                        A, B, C, r0, step)
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
    data : numpy.ndarray
        Array containing the data.
    sigma : float
        Standard deviation of noise in each channel.
    seed : int, optional
        Seed for pseudorandom number generation.

    Returns
    -------
    noisy_data : numpy.ndarray
        Noisy data.
    """
    np.random.seed(seed)
    noisy_data = np.abs(
        data + np.random.normal(size=data.shape, scale=sigma, loc=0)
        + 1j * np.random.normal(size=data.shape, scale=sigma, loc=0))
    return noisy_data


def simulation(n_spins, diffusivity, gradient, dt, substrate, seed=123,
               trajectories=None, final_pos=False, all_signals=False,
               quiet=False, cuda_bs=128, max_iter=int(1e3)):
    """Execute a dMRI simulation. For a detailed tutorial, please see the
    documentation at https://disimpy.readthedocs.io/en/latest/tutorial.html.

    Parameters
    ----------
    n_spins : int
        Number of random walkers.
    diffusivity : float
        Diffusivity in SI units (m^2/s).
    gradient : numpy.ndarray
        Gradient array of shape (n of measurements, n of time points, 3). Array
        elements are floats representing the gradient magnitude at that time
        point in SI units (T/m).
    dt : float
        Duration of a time step in the gradient array in SI units (s).
    substrate : dict
        A dictionary defining the diffusion environment.
    seed : int, optional
        Seed for pseudorandom number generation.
    trajectories : str, optional
        Path of a file in which to save the simulated trajectories. Resulting
        file can be very large!
    final_pos : bool, optional
        If true, the function returns the signal and the final positions of the
        walkers at the end of the simulation.
    all_signals : bool, optional
        If true, the function returns the signals from each walker instead of
        total signal.
    quiet : bool, optional
        Whether to print messages about simulation progression.
    cuda_bs : int, optional
        The size of the CUDA thread block (1D).
    max_iter : int, optional
        The maximum number of allowed iterations in the intersection check
        algorithm.

    Returns
    -------
    signal : numpy.ndarray
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
        raise ValueError(
            'Incorrect value (%s) for parameter gradient.' %
            gradient +
            ' Gradient array must be a floating point array of' +
            ' shape (n of measurements, n of time points, 3).')
    if not (isinstance(dt, int) or isinstance(dt, float)) or (dt <= 0):
        raise ValueError(
            'Incorrect value (%s) for parameter dt which has to' %
            dt + ' be a positive integer or float.')
    if (not isinstance(substrate, dict)) or ('type' not in substrate.keys()):
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
            raise ValueError(
                'Incorrect value (%s) for parameter' %
                trajectories +
                ' trajectories which has to be a string.')
    if not isinstance(quiet, bool):
        raise ValueError('Incorrect value (%s) for parameter quiet' % quiet
                         + ' which has to be a boolean.')
    if (not isinstance(cuda_bs, int)) or (cuda_bs <= 0):
        raise ValueError('Incorrect value (%s) for parameter cuda_bs' % cuda_bs
                         + ' which has to be a positive integer.')
    if (not isinstance(max_iter, int)) or (max_iter <= 0):
        raise ValueError(
            'Incorrect value (%s) for parameter max_iter' % max_iter
            + ' which has to be a positive integer.')

    if not quiet:
        print('Starting simulation')
        if trajectories:
            print('The trajectories file will be up to %s GB'
                  % (gradient.shape[1] * n_spins * 3 * 25 / 1e9))

    # Set up cuda stream
    bs = cuda_bs  # Threads per block
    gs = int(math.ceil(float(n_spins) / bs))  # Blocks per grid
    stream = cuda.stream()

    global MAX_ITER
    MAX_ITER = max_iter

    # Create pseudorandom number generator states
    rng_states = create_xoroshiro128p_states(gs * bs, seed=seed, stream=stream)

    # Move required arrays to the GPU
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
        print('Number of spins = %s' % n_spins)
        print('Number of steps = %s' % gradient.shape[1])
        print('Step length = %s m' % step_l)
        print('Step duration = %s s' % dt)

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
        for t in range(gradient.shape[1]):
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
        if (('radius' not in substrate.keys()) or
                ('orientation' not in substrate.keys())):
            raise ValueError('Incorrect value (%s) for parameter' % substrate
                             + ' substrate which has to be a dictionary with'
                             + ' keys \'radius\' and \'orientation\' when'
                             + ' simulating diffusion inside an infinite'
                             + ' cylinder.')
        radius = substrate['radius']
        if (not isinstance(radius, float)) or (radius <= 0):
            raise ValueError(
                'Incorrect value (%s) for cylinder radius' %
                radius + ' which has to be a positive float.')
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
        if 'radius' not in substrate.keys():
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
        if (('a' not in substrate.keys()) or ('b' not in substrate.keys())
                or ('c' not in substrate.keys()) or
                ('R' not in substrate.keys())):
            raise ValueError('Incorrect value (%s) for parameter' % substrate
                             + ' substrate which has to be a dictionary with'
                             + ' keys \'a\', \'b\', \'c\' and \'R\' when'
                             + ' simulating diffusion inside an ellipsoid.')
        a = substrate['a']
        if (not isinstance(a, float)) or (a <= 0):
            raise ValueError(
                'Incorrect value (%s) for ellipsoid semiaxis a' %
                a + ' which has to be a positive float.')
        b = substrate['b']
        if (not isinstance(b, float)) or (b <= 0):
            raise ValueError(
                'Incorrect value (%s) for ellipsoid semiaxis b' %
                b + ' which has to be a positive float.')
        c = substrate['c']
        if (not isinstance(c, float)) or (c <= 0):
            raise ValueError(
                'Incorrect value (%s) for ellipsoid semiaxis c' %
                c + ' which has to be a positive float.')
        R = substrate['R']
        if ((not isinstance(R, np.ndarray)) or (R.shape != (3, 3)) or
                (R.dtype != float)):
            raise ValueError(
                'Incorrect value (%s) for rotation matrix R' %
                R + ' which has to be a float array of shape (3, 3).')

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
            _cuda_step_ellipsoid[gs,
                                 bs,
                                 stream](d_positions,
                                         d_g_x,
                                         d_g_y,
                                         d_g_z,
                                         d_phases,
                                         rng_states,
                                         t,
                                         GAMMA,
                                         step_l,
                                         dt,
                                         a,
                                         b,
                                         c,
                                         R,
                                         R_inv,
                                         d_iter_exc)
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
        if 'mesh' not in substrate.keys():
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
        if ('intra' not in substrate) and ('extra' not in substrate):
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
        if 'periodic' in substrate:
            periodic = substrate['periodic']
            print('Periodic boundary conditions is an experimental feature.')
            if not isinstance(periodic, bool):
                raise ValueError('Incorrect value (%s) for periodic' % periodic
                                 + ' which has to be boolean.')
        else:
            periodic = False

        # Align the corner of the mesh with the origin of the coordinate system
        mesh -= np.min(np.min(mesh, 0), 0)

        # Calculate subvoxel division
        if not quiet:
            print("Calculating subvoxel division.", end="\r")
        sv_borders = _mesh_space_subdivision(mesh, N=N_sv)
        if step_l > min(np.max(np.max(mesh, 0), 0)):
            raise ValueError('Step length is too long for good results. ' +
                             'Please increase number of time steps or lower' +
                             'diffusivity.')
        tri_indices, sv_mapping = _subvoxel_to_triangle_mapping(
            mesh, sv_borders)
        d_sv_borders = cuda.to_device(sv_borders, stream=stream)
        d_tri_indices = cuda.to_device(tri_indices, stream=stream)
        d_sv_mapping = cuda.to_device(sv_mapping, stream=stream)

        # Calculate initial positions
        if 'initial positions' in substrate:
            if not quiet:
                print('Initialized random walker positions.')
            positions = substrate['initial positions']
            if (not isinstance(positions, np.ndarray) or
                positions.shape != (n_spins, 3) or
                    positions.dtype != np.float):
                raise ValueError('Incorrect value for initial positions which'
                                 + 'has to be a float array of shape (n of '
                                 + 'spins, 3).')
        else:
            if not quiet:
                print("Calculating initial positions.", end="\r")
            positions = _fill_mesh(n_spins, mesh, sv_borders,
                                   tri_indices, sv_mapping, intra, extra)
            if not quiet:
                print("Finished calculating initial positions.")
        if trajectories:
            with open(trajectories, 'w') as f:
                [f.write(str(i) + ' ') for i in positions.ravel()]
                f.write('\n')
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Calculate voxel boundaries as triangular mesh
        voxel_mesh = _AABB_to_mesh(np.min(np.min(mesh, 0), 0),
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
                                            d_iter_exc, periodic)
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
        warn('Maximum number of iterations was exceeded in the intersection ' +
             'check algorithm for walkers %s.' % np.where(iter_exc)[0])

    # Calculate simulated signal
    if not quiet:
        print('Simulation finished.')
    if all_signals:  # Return signals from individual walkers
        phases = d_phases.copy_to_host(stream=stream)
        phases[:, np.where(iter_exc)[0]] = np.nan
        signals = np.real(np.exp(1j * phases))
    else:
        phases = d_phases.copy_to_host(stream=stream)
        phases[:, np.where(iter_exc)[0]] = np.nan
        signals = np.real(np.nansum(np.exp(1j * phases), axis=1))
    if final_pos:  # Return final positions
        positions = d_positions.copy_to_host(stream=stream)
        return signals, positions
    else:
        return signals
