"""This module contains code for executing diffusion-weighted MR simulations."""

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

from . import utils, substrates
from .gradients import GAMMA


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
    for i in range(3):
        step[i] = xoroshiro128p_normal_float64(rng_states, thread_id)
    _cuda_normalize_vector(step)
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
    rotated_v = cuda.local.array(3, numba.float64)
    rotated_v[0] = R[0, 0] * v[0] + R[0, 1] * v[1] + R[0, 2] * v[2]
    rotated_v[1] = R[1, 0] * v[0] + R[1, 1] * v[1] + R[1, 2] * v[2]
    rotated_v[2] = R[2, 0] * v[0] + R[2, 1] * v[1] + R[2, 2] * v[2]
    for i in range(3):
        v[i] = rotated_v[i]
    return


@cuda.jit(device=True)
def _cuda_line_circle_intersection(r0, step, radius):
    """Calculate the distance from r0 to a circle centered at origin along step.
    r0 must be inside the circle.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    radius : float

    Returns
    -------
    float
    """
    A = step[0]**2 + step[1]**2
    B = 2 * (r0[0] * step[0] + r0[1] * step[1])
    C = r0[0]**2 + r0[1]**2 - radius**2
    d = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return d


@cuda.jit(device=True)
def _cuda_line_sphere_intersection(r0, step, radius):
    """Calculate the distance from r0 to a sphere centered at origin along step.
    r0 must be inside the sphere.

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
    A = (step[0] / a)**2 + (step[1] / b)**2 + (step[2] / c)**2
    B = 2 * (a**(-2) * step[0] * r0[0] + b**(-2) *
             step[1] * r0[1] + c**(-2) * step[2] * r0[2])
    C = (r0[0] / a)**2 + (r0[1] / b)**2 + (r0[2] / c)**2 - 1
    d = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
    return d


@numba.jit(nopython=True)
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
    intersection = cuda.local.array(3, numba.float64)
    v = cuda.local.array(3, numba.float64)
    for i in range(3):
        intersection[i] = r0[i] + d * step[i]
        v[i] = intersection[i] - r0[i]
    dp = _cuda_dot_product(v, normal)
    if dp > 0:  # Make sure the normal vector points against the step
        for i in range(3):
            normal[i] *= -1
        dp = _cuda_dot_product(v, normal)
    for i in range(3):
        step[i] = (
            (v[i] - 2 * dp * normal[i] + intersection[i]) - intersection[i])
    _cuda_normalize_vector(step)
    for i in range(3):  # Move walker slightly away from the surface
        r0[i] = intersection[i] + epsilon * normal[i]
    return


@cuda.jit(device=True)
def _cuda_ray_triangle_intersection_check(triangle, r0, step):
    """Check if a ray defined by r0 and step intersects with a triangle defined
    by A, B, and C. The output is the distance in units of step length from r0
    to intersection if intersection found, nan otherwise. This function is based
    on the Moller-Trumbore algorithm.

    Parameters
    ----------
    A : numba.cuda.cudadrv.devicearray.DeviceNDArray
    B : numba.cuda.cudadrv.devicearray.DeviceNDArray
    C : numba.cuda.cudadrv.devicearray.DeviceNDArray
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    epsilon : float

    Returns
    -------
    float
    """
    A = triangle[0]
    B = triangle[1]
    C = triangle[2]
    T = cuda.local.array(3, numba.float64)
    E_1 = cuda.local.array(3, numba.float64)
    E_2 = cuda.local.array(3, numba.float64)
    P = cuda.local.array(3, numba.float64)
    Q = cuda.local.array(3, numba.float64)
    for i in range(3):
        T[i] = r0[i] - A[i]
        E_1[i] = B[i] - A[i]
        E_2[i] = C[i] - A[i]
    _cuda_cross_product(step, E_2, P)
    _cuda_cross_product(T, E_1, Q)
    det = _cuda_dot_product(P, E_1)
    #if abs(det) > 1e-7:
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


def _mesh_space_subdivision(triangles, voxel_size, n_sv):
    """Divide a triangular mesh into into subvoxels and return arrays for
    finding the relevant triangles of given a subvoxel.   

    Parameters
    ----------
    triangles : numpy.ndarray
        Triangular mesh represented by an array of shape (number of triangles,
        3, 3) where the second dimension indices correspond to different
        triangle points and the third dimension representes the Cartesian
        coordinates.
    voxel_size : numpy.ndarray
        Floating-point array of size (3,) defining the size of the simulated
        voxel.
    n_sv : numpy.ndarray
        Integer array of size (3,) defining the number of subvoxels along each
        axis.

    Returns
    -------
    xs : numpy.ndarray
        Floating-point array of shape (n_sv[0],) defining the subvoxel
        boundaries along the x-axis.
    ys : numpy.ndarray
        Floating-point array of shape (n_sv[1],) defining the subvoxel
        boundaries along the y-axis.
    zs : numpy.ndarray
        Floating-point array of shape (n_sv[2],) defining the subvoxel
        boundaries along the z-axis.
    triangle_indices : numpy.ndarray
        One-dimensional integer array ontaining the relevant triangle indices
        for all subvoxels.
    subvoxel_indices : numpy.ndarray
        Two-dimensional integer array that enables the relevant triangles to a
        given subvoxel to be located in the array triangle_indices. The
        relevant triangles to subvoxel i are the elements of tri_indices from
        subvoxel_indices[i, 0] to subvoxel_indices[i, 1].

    Notes
    -----
    This implementation finds the relevant triangles by comparing the axis
    aligned bounding boxes of the triangle to the subvoxel grid. The output is
    two arrays so that it can be passed onto CUDA kernels which do not support
    3D arrays or Python data structures like lists of lists or dictionaries.
    """

    # Define the subvoxel boundaries
    n_sv = np.array([n_sv, n_sv, n_sv])  # TEMPORARILY
    xs = np.linspace(0, voxel_size[0], n_sv[0] + 1)
    ys = np.linspace(0, voxel_size[1], n_sv[1] + 1)
    zs = np.linspace(0, voxel_size[2], n_sv[2] + 1)
    relevant_triangles = [[] for _ in range(np.product(n_sv))]
 
    # Loop over the triangles to assign them to subvoxels
    for i, triangle in enumerate(triangles):
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
                    subvoxel = x * n_sv[1] * n_sv[2] + y * n_sv[1] + z
                    relevant_triangles[subvoxel].append(i)
    
    # Make the final arrays
    triangle_indices = []
    subvoxel_indices = np.zeros((len(relevant_triangles), 2))
    counter = 0
    for i, l in enumerate(relevant_triangles):
        triangle_indices += l
        subvoxel_indices[i, 0] = counter
        subvoxel_indices[i, 1] = counter + len(l)
        counter += len(l)
    triangle_indices = np.array(triangle_indices).astype(int)
    subvoxel_indices = subvoxel_indices.astype(int)
    return xs, ys, zs, triangle_indices, subvoxel_indices


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
def _fill_ellipsoid(n, semiaxes, seed=123):
    """Sample n random points from a uniform distribution inside an axis aligned
    ellipsoid with semi-axes a, b, and c."""
    np.random.seed(seed)
    filled = False
    points = np.zeros((n, 3))
    i = 0
    while not filled:
        p = (np.random.random(3) - .5) * 2 * semiaxes
        if np.sum((p / semiaxes)**2) < 1:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


def _initial_positions_cylinder(n_walkers, radius, R, seed=123):
    """Calculate initial positions for spins in a cylinder whose orientation is
    defined by R which defines the rotation from cylinder frame to lab frame."""
    positions = np.zeros((n_walkers, 3))
    positions[:, 1:3] = _fill_circle(n_walkers, radius, seed)
    positions = np.matmul(R, positions.T).T
    return positions


def _initial_positions_ellipsoid(n_walkers, semiaxes, R, seed=123):
    """Calculate initial positions for spins in an ellipsoid with semi-axes a,
    b, c whos whose orientation is defined by R which defines the rotation from
    ellipsoid frame to lab frame."""
    positions = _fill_ellipsoid(n_walkers, semiaxes, seed)
    positions = np.matmul(R, positions.T).T
    return positions


@cuda.jit()
def _cuda_fill_mesh(points, rng_states, vertices, faces, voxel_size, intra):
    """Sample points from a uniform distribution inside or outside the surface
    defined by a triangular mesh.

    Parameters
    ----------
    points : numba.cuda.cudadrv.devicearray.DeviceNDArray
    rng_states : numba.cuda.cudadrv.devicearray.DeviceNDArray
    vertices : numba.cuda.cudadrv.devicearray.DeviceNDArray
    faces : numba.cuda.cudadrv.devicearray.DeviceNDArray
    voxel_size : numba.cuda.cudadrv.devicearray.DeviceNDArray
    intra : bool

    Returns
    -------
    None
    """
    thread_id = cuda.grid(1)
    if thread_id >= points.shape[0] or points[thread_id, 0] != math.inf:
        return
    point = cuda.local.array(3, numba.float64)
    for i in range(3):
        point[i] = xoroshiro128p_uniform_float64(
            rng_states, thread_id) * voxel_size[i]
    ray = cuda.local.array(3, numba.float64)
    _cuda_random_step(ray, rng_states, thread_id)
    intersections = 0
    triangle = cuda.local.array((3, 3), numba.float64)
    for idx in faces:  # Loop over triangles
        for i in range(3):
            for j in range(3):
                triangle[i, j] = vertices[idx[i], j]
        t = _cuda_ray_triangle_intersection_check(triangle, point, ray)
        if t > 0:
            intersections += 1
    if intra:
        if intersections % 2 == 1:  # Point is inside the surface
            for i in range(3):
                points[thread_id, i] = point[i]
    else:
        if intersections % 2 == 0:  # Point is outside the surface
            for i in range(3):
                points[thread_id, i] = point[i]
    return


def _fill_mesh(n_points, substrate, intra, seed=123, cuda_bs=128):
    """Sample points from a uniform distribution inside or outside the surface
    defined by a triangular mesh.

    Parameters
    ----------
    n_walkers : np.ndarray
    substrate : substrates._Substrate
    intra : bool
    seed : int, optional
    cuda_bs : int, optional

    Returns
    -------
    points : np.ndarray
    """
    bs = cuda_bs
    gs = int(math.ceil(float(n_points) / bs))
    stream = cuda.stream()
    rng_states = create_xoroshiro128p_states(gs * bs, seed=seed, stream=stream)
    points = np.ones((n_points, 3)).astype(np.float64) * math.inf
    d_points = cuda.to_device(points, stream=stream)
    if substrate.periodic:
        d_vertices = cuda.to_device(substrate.vertices, stream=stream)
        d_faces = cuda.to_device(substrate.faces, stream=stream)
    else:  # Don't include the voxel boundaries
        d_vertices = cuda.to_device(substrate.vertices[0:-8], stream=stream)
        d_faces = cuda.to_device(substrate.faces[0:-12], stream=stream)
    d_voxel_size = cuda.to_device(substrate.voxel_size, stream=stream)
    while np.any(np.isinf(points)):
        _cuda_fill_mesh[gs, bs, stream](
            d_points, rng_states, d_vertices, d_faces, d_voxel_size, intra)
        stream.synchronize()
        points = d_points.copy_to_host(stream=stream)
    return points


def _aabb_to_mesh(a, b):
    """Return a triangular mesh that corresponds to an axis-aligned bounding box
    defined by points a and b."""
    vertices = np.array([[a[0], a[1], a[2]],
                         [b[0], a[1], a[2]],
                         [b[0], b[1], a[2]],
                         [b[0], b[1], b[2]],
                         [a[0], b[1], b[2]],
                         [a[0], a[1], b[2]],
                         [a[0], b[1], a[2]],
                         [b[0], a[1], b[2]]])
    faces = np.array([[0, 1, 2],
                      [0, 6, 2],
                      [5, 7, 3],
                      [5, 4, 3],
                      [1, 2, 3],
                      [1, 7, 3],
                      [0, 6, 4],
                      [0, 5, 4],
                      [0, 1, 7],
                      [0, 5, 7],
                      [6, 2, 3],
                      [6, 4, 3]])
    return vertices, faces


@cuda.jit(device=True)
def _ll_subvoxel_overlap_1d(xs, x1, x2):
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
def _ul_subvoxel_overlap_1d(xs, x1, x2):
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
def _cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, t, step_l,
                    dt):
    """Kernel function for free diffusion."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.float64)
    _cuda_random_step(step, rng_states, thread_id)
    for i in range(3):
        positions[thread_id, i] = positions[thread_id, i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (GAMMA * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_sphere(positions, g_x, g_y, g_z, phases, rng_states, t, step_l,
                      dt, radius, iter_exc, max_iter, epsilon):
    """Kernel function for diffusion inside a sphere."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.float64)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    iter_idx = 0
    check_intersection = True
    while check_intersection and step_l > 0 and iter_idx < max_iter:
        iter_idx += 1
        d = _cuda_line_sphere_intersection(r0, step, radius)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.float64)
            for i in range(3):
                normal[i] = -(r0[i] + d * step[i])
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal, epsilon)
            step_l -= d + epsilon
        else:
            check_intersection = False
    if iter_idx >= max_iter:
        iter_exc[thread_id] = True
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (GAMMA * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_cylinder(positions, g_x, g_y, g_z, phases, rng_states, t, step_l,
                        dt, radius, R, R_inv, iter_exc, max_iter, epsilon):
    """Kernel function for diffusion inside an infinite cylinder."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.float64)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    _cuda_mat_mul(R, r0)  # Move to cylinder frame
    iter_idx = 0
    check_intersection = True
    while check_intersection and step_l > 0 and iter_idx < max_iter:
        iter_idx += 1
        d = _cuda_line_circle_intersection(r0[1:3], step[1:3], radius)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.float64)
            normal[0] = 0
            for i in range(1, 3):
                normal[i] = -(r0[i] + d * step[i])
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal, epsilon)
            step_l -= d + epsilon
        else:
            check_intersection = False
    if iter_idx >= max_iter:
        iter_exc[thread_id] = True
    _cuda_mat_mul(R_inv, step)  # Move back to lab frame
    _cuda_mat_mul(R_inv, r0)  # Move back to lab frame
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (GAMMA * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_ellipsoid(positions, g_x, g_y, g_z, phases, rng_states, t,
                         step_l, dt, semiaxes, R, R_inv, iter_exc, max_iter,
                         epsilon):
    """Kernel function for diffusion inside an ellipsoid."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.float64)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    _cuda_mat_mul(R, r0)  # Move to ellipsoid frame
    iter_idx = 0
    check_intersection = True
    while check_intersection and step_l > 0 and iter_idx < max_iter:
        iter_idx += 1
        d = _cuda_line_ellipsoid_intersection(r0, step, semiaxes)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.float64)
            for i in range(3):
                normal[i] = -(r0[i] + d * step[i]) / semiaxes[i]**2
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal, epsilon)
            step_l -= d + epsilon
        else:
            check_intersection = False
    if iter_idx >= max_iter:
        iter_exc[thread_id] = True
    _cuda_mat_mul(R_inv, step)  # Move back to lab frame
    _cuda_mat_mul(R_inv, r0)  # Move back to lab frame
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (GAMMA * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


@cuda.jit()
def _cuda_step_mesh(positions, g_x, g_y, g_z, phases, rng_states, t, step_l, dt,
                    vertices, faces, xs, ys, zs, subvoxel_indices,
                    triangle_indices, iter_exc, max_iter, n_sv, epsilon):
    """Kernel function for diffusion restricted by a triangular mesh."""

    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return

    # Allocate memory
    step = cuda.local.array(3, numba.float64)
    triangle = cuda.local.array((3, 3), numba.float64)
    lls = cuda.local.array(3, numba.int64)
    uls = cuda.local.array(3, numba.int64)

    # Get position and generate step
    r0 = positions[thread_id, :]
    _cuda_random_step(step, rng_states, thread_id)


    # Find the relevant subvoxels for the step
    lls[0] = _ll_subvoxel_overlap_1d(xs, r0[0], r0[0] + step[0] * step_l)
    lls[1] = _ll_subvoxel_overlap_1d(ys, r0[1], r0[1] + step[1] * step_l)
    lls[2] = _ll_subvoxel_overlap_1d(zs, r0[2], r0[2] + step[2] * step_l)
    uls[0] = _ul_subvoxel_overlap_1d(xs, r0[0], r0[0] + step[0] * step_l)
    uls[1] = _ul_subvoxel_overlap_1d(ys, r0[1], r0[1] + step[1] * step_l)
    uls[2] = _ul_subvoxel_overlap_1d(zs, r0[2], r0[2] + step[2] * step_l)

    # Loop over subvoxels and fnd the closest triangle
    for x in range(lls[0], uls[0]):
        for y in range(lls[1], uls[1]):
            for z in range(lls[2], uls[2]):
                subvoxel = int(x * n_sv[1] * n_sv[2] + y * n_sv[2] + z)

                # Loop over the triangles in this subvoxel
                for a in range(subvoxel_indices[subvoxel, 0],
                               subvoxel_indices[subvoxel, 1]):
                    idx = faces[triangle_indices[a]]
                    for i in range(3):
                        for j in range(3):
                            triangle[i, j] = vertices[idx[i], j]
                    d = _cuda_ray_triangle_intersection_check(
                        triangle, r0, step)
                    if d > 0 and d < step_l:
                        step_l = 0


    #for idx in faces:  # Loop over triangles
    #    for i in range(3):
    #        for j in range(3):
    #            triangle[i, j] = vertices[idx[i], j]
    #    d = _cuda_ray_triangle_intersection_check(triangle, r0, step)
    #    if d > 0 and d < step_l:
    #        step_l = 0

    
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (GAMMA * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return


"""
@cuda.jit()
def _cuda_step_mesh(positions, g_x, g_y, g_z, phases, rng_states, t, step_l, dt,
                    triangles, xs, ys, zs, subvoxel_indices, triangle_indices,
                    iter_exc, max_iter, periodic, n_sv, epsilon):
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.float64)
    _cuda_random_step(step, rng_states, thread_id)
    r0 = cuda.local.array(3, numba.float64)
    r0 = positions[thread_id, :]
    iter_idx = 0
    prev_tri_idx = -1
    check_intersection = True
    shifts = cuda.local.array(3, numba.float64)
    temp_shifts = cuda.local.array(3, numba.float64)
    temp_r0 = cuda.local.array(3, numba.float64)
    lls = cuda.local.array(3, numba.float64)
    uls = cuda.local.array(3, numba.float64)
    while check_intersection and step_l > 0 and iter_idx < max_iter:
        iter_idx += 1
        min_d = math.inf
        min_idx = 0
        
        lls[0] = ll_subvoxel_overlap_1d_periodic(
            xs, r0[0], r0[0] + step[0] * step_l)
        uls[0] = ul_subvoxel_overlap_1d_periodic(
            xs, r0[0], r0[0] + step[0] * step_l)
        lls[1] = ll_subvoxel_overlap_1d_periodic(
            ys, r0[1], r0[1] + step[1] * step_l)
        uls[1] = ul_subvoxel_overlap_1d_periodic(
            ys, r0[1], r0[1] + step[1] * step_l)
        lls[2] = ll_subvoxel_overlap_1d_periodic(
            zs, r0[2], r0[2] + step[2] * step_l)
        uls[2] = ul_subvoxel_overlap_1d_periodic(
            zs, r0[2], r0[2] + step[2] * step_l)

        for x in range(lls[0], uls[0]):  # Loop over relevant subvoxels
            if x < 0 or x > n_sv - 1:
                shift_n = math.floor(x / n_sv)
                x -= shift_n * n_sv
                temp_shifts[0] = shift_n * xs[-1]
            else:
                temp_shifts[0] = 0
            for y in range(lls[1], uls[1]):
                if y < 0 or y > n_sv - 1:
                    shift_n = math.floor(y / n_sv)
                    y -= shift_n * n_sv
                    temp_shifts[1] = shift_n * ys[-1]
                else:
                    temp_shifts[1] = 0
                for z in range(lls[2], uls[2]):
                    if z < 0 or z > n_sv - 1:
                        shift_n = math.floor(z / n_sv)
                        z -= shift_n * n_sv
                        temp_shifts[2] = shift_n * zs[-1]
                    else:
                        temp_shifts[2] = 0
                    sv_idx = int(x * n_sv**2 + y * n_sv + z)
                    for i in range(3):  # Shift walker
                        temp_r0[i] = r0[i] - temp_shifts[i]
                    # Loop over relevant triangles
                    for i in range(subvoxel_indices[sv_idx, 0],
                                   subvoxel_indices[sv_idx, 1]):
                        tri_idx = triangle_indices[i] * 9
                        if tri_idx != prev_tri_idx:
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
            normal = cuda.local.array(3, numba.float64)
            normal[0] = ((B[1] - A[1]) * (C[2] - A[2]) -
                         (B[2] - A[2]) * (C[1] - A[1]))
            normal[1] = ((B[2] - A[2]) * (C[0] - A[0]) -
                         (B[0] - A[0]) * (C[2] - A[2]))
            normal[2] = ((B[0] - A[0]) * (C[1] - A[1]) -
                         (B[1] - A[1]) * (C[0] - A[0]))
            _cuda_normalize_vector(normal)
            for i in range(3):  # Shift walker to voxel
                r0[i] -= shifts[i]
            _cuda_reflection(r0, step, min_d, normal, epsilon)
            for i in range(3):  # Shift walker back
                r0[i] += shifts[i]
            step_l -= min_d + epsilon
            prev_tri_idx = min_idx
        else:
            check_intersection = False
    
    if iter_idx >= max_iter:
        iter_exc[thread_id] = True
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (GAMMA * dt *
                                 ((g_x[m, t] * positions[thread_id, 0])
                                  + (g_y[m, t] * positions[thread_id, 1])
                                  + (g_z[m, t] * positions[thread_id, 2])))
    return
"""


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


def _write_traj(traj, mode, positions):
    """Write random walker trajectories to a file."""
    with open(traj, mode) as f:
        [f.write(str(i) + ' ') for i in positions.ravel()]
        f.write('\n')
    return


def simulation(n_walkers, diffusivity, gradient, dt, substrate, seed=123,
               traj=None, final_pos=False, all_signals=False,
               quiet=False, cuda_bs=128, max_iter=int(1e3), epsilon=1e-10):
    """Execute a dMRI simulation. For a detailed tutorial, please see the
    documentation at https://disimpy.readthedocs.io/en/latest/tutorial.html.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers.
    diffusivity : float
        Diffusivity in SI units (m^2/s).
    gradient : numpy.ndarray
        Array of shape (number of measurements, number of time points, 3). Array
        elements are floating-point numbers representing the gradient magnitude
        at that time point in SI units (T/m).
    dt : float
        Duration of a time step in the gradient array in SI units (s).
    substrate : disimpy.substrates._Substrate
        Substrate object that contains information about the simulated
        microstructure.
    seed : int, optional
        Seed for pseudorandom number generation.
    traj : str, optional
        Path of a file in which to save the simulated random walker
        trajectories. The file can become very large!
    final_pos : bool, optional
        If True, the signal and the final positions of the random walkers are
        returned.
    all_signals : bool, optional
        If True, the signals from each walker are returned instead of the total
        signal.
    quiet : bool, optional
        If True, updates on the progress of the simulation are not printed.
    cuda_bs : int, optional
        The size of the one-dimensional CUDA thread block.
    max_iter : int, optional
        The maximum number of iterations in the algorithm that checks if a
        random walker collides with a surface during a time step.
    epsilon : float, optional
        The amount by which a random walker is moved away from the surface after
        a collision to avoid placing it in the surface.

    Returns
    -------
    signal : numpy.ndarray
        Simulated signals.
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

    # Validate input
    if not isinstance(n_walkers, int) or n_walkers <= 0:
        raise ValueError('Incorrect value (%s) for n_walkers' % n_walkers)
    if not isinstance(diffusivity, float) or diffusivity <= 0:
        raise ValueError('Incorrect value (%s) for diffusivity' % diffusivity)
    if (not isinstance(gradient, np.ndarray) or gradient.ndim != 3 or
        gradient.shape[2] != 3 or gradient.dtype != float):
        raise ValueError('Incorrect value (%s) for gradient' % gradient)
    if not isinstance(dt, float) or dt <= 0:
        raise ValueError('Incorrect value (%s) for dt' % dt)
    if not isinstance(substrate, substrates._Substrate):
        raise ValueError('Incorrect value (%s) for substrate' % substrate)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Incorrect value (%s) for seed' % seed)
    if traj:
        if not isinstance(traj, str):
            raise ValueError(
                'Incorrect value (%s) for traj' % traj)
    if not isinstance(quiet, bool):
        raise ValueError('Incorrect value (%s) for quiet' % quiet)
    if not isinstance(cuda_bs, int) or cuda_bs <= 0:
        raise ValueError('Incorrect value (%s) for cuda_bs' % cuda_bs)
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('Incorrect value (%s) for max_iter' % max_iter)

    if not quiet:
        print('Starting simulation')
        if traj:
            print('The trajectories file will be up to %s GB'
                  % (gradient.shape[1] * n_walkers * 3 * 25 / 1e9))

    # Set up CUDA stream
    bs = cuda_bs  # Threads per block
    gs = int(math.ceil(float(n_walkers) / bs))  # Blocks per grid
    stream = cuda.stream()

    # Set seed and create PRNG states
    np.random.seed(seed)
    rng_states = create_xoroshiro128p_states(gs * bs, seed=seed, stream=stream)

    # Move arrays to the GPU
    d_g_x = cuda.to_device(
        np.ascontiguousarray(gradient[:, :, 0]), stream=stream)
    d_g_y = cuda.to_device(
        np.ascontiguousarray(gradient[:, :, 1]), stream=stream)
    d_g_z = cuda.to_device(
        np.ascontiguousarray(gradient[:, :, 2]), stream=stream)
    d_iter_exc = cuda.to_device(np.zeros(n_walkers).astype(bool))
    d_phases = cuda.to_device(
        np.ascontiguousarray(np.zeros((gradient.shape[0], n_walkers))),
        stream=stream)

    step_l = np.sqrt(6 * diffusivity * dt)

    if not quiet:
        print('Number of random walkers = %s' % n_walkers)
        print('Number of steps = %s' % gradient.shape[1])
        print('Step length = %s m' % step_l)
        print('Step duration = %s s' % dt)

    if substrate.type == 'free':

        # Define initial positions
        positions = np.zeros((n_walkers, 3))
        if traj:
            _write_traj(traj, 'w', positions)
        d_positions = cuda.to_device(positions, stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_free[gs, bs, stream](
                d_positions, d_g_x, d_g_y, d_g_z, d_phases, rng_states, t,
                step_l, dt)
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, 'a', positions)
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate.type == 'cylinder':

        radius = substrate.radius
        orientation = substrate.orientation

        # Calculate rotation from lab frame to cylinder frame and back
        default_orientation = np.array([1., 0, 0])
        R = utils.vec2vec_rotmat(orientation, default_orientation)
        R_inv = np.linalg.inv(R)

        # Calculate initial positions
        positions = _initial_positions_cylinder(n_walkers, radius, R_inv, seed)
        if traj:
            _write_traj(traj, 'w', positions)
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_cylinder[gs, bs, stream](
                d_positions, d_g_x, d_g_y, d_g_z, d_phases, rng_states, t,
                step_l, dt, radius, R, R_inv, d_iter_exc, max_iter, epsilon)
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, 'a', positions)
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate.type == 'sphere':

        radius = substrate.radius

        # Calculate initial positions
        positions = _fill_sphere(n_walkers, radius, seed)
        if traj:
            _write_traj(traj, 'w', positions)
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_sphere[gs, bs, stream](
                d_positions, d_g_x, d_g_y, d_g_z, d_phases, rng_states, t,
                step_l, dt, radius, d_iter_exc, max_iter, epsilon)
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, 'a', positions)
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate.type == 'ellipsoid':

        semiaxes = substrate.semiaxes
        d_semiaxes = cuda.to_device(np.ascontiguousarray(semiaxes))

        # Calculate rotation from ellipsoid frame to lab frame and back
        R_inv = substrate.R
        R = np.linalg.inv(R_inv)

        # Calculate initial positions
        positions = _initial_positions_ellipsoid(n_walkers, semiaxes, R_inv)
        if traj:
            _write_traj(traj, 'w', positions)
        d_positions = cuda.to_device(
            np.ascontiguousarray(positions), stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_ellipsoid[gs, bs, stream](
                d_positions, d_g_x, d_g_y, d_g_z, d_phases, rng_states, t,
                step_l, dt, d_semiaxes, R, R_inv, d_iter_exc, max_iter, epsilon)
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, 'a', positions)
            if not quiet:
                print(
                    str(np.round((t / gradient.shape[1]) * 100, 0)) + ' %',
                    end="\r")

    elif substrate.type == 'mesh':

        # Calculate initial positions
        if isinstance(substrate.init_pos, np.ndarray):
            if n_walkers != substrate.init_pos.shape[0]:
                raise ValueError(
                    'n_walkers must be equal to the number of initial positions'
                )
            positions = substrate.init_pos
            
        else:
            if not quiet:
                print('Calculating initial positions')
            if substrate.init_pos == 'uniform':
                positions = (np.random.random((n_walkers, 3))
                             * substrate.voxel_size)
            elif substrate.init_pos == 'intra':
                positions = _fill_mesh(n_walkers, substrate, True)
            else:
                positions = _fill_mesh(n_walkers, substrate, False)
            if not quiet:
                print('Finished calculating initial positions')
        if traj:
            _write_traj(traj, 'w', positions)
        
        # Move arrays to the GPU
        d_vertices = cuda.to_device(substrate.vertices, stream=stream)
        d_faces = cuda.to_device(substrate.faces, stream=stream)
        d_xs = cuda.to_device(substrate.xs, stream=stream)
        d_ys = cuda.to_device(substrate.ys, stream=stream)
        d_zs = cuda.to_device(substrate.zs, stream=stream)
        d_triangle_indices = cuda.to_device(substrate.triangle_indices, stream=stream)
        d_subvoxel_indices = cuda.to_device(substrate.subvoxel_indices, stream=stream)
        d_voxel_size = cuda.to_device(substrate.voxel_size, stream=stream)
        d_n_sv = cuda.to_device(substrate.n_sv, stream=stream)
        d_positions = cuda.to_device(positions, stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_mesh[gs, bs, stream](
                d_positions, d_g_x, d_g_y, d_g_z, d_phases, rng_states, t,
                step_l, dt, d_vertices, d_faces, d_xs, d_ys, d_zs,
                d_subvoxel_indices, d_triangle_indices, d_iter_exc, max_iter,
                d_n_sv, epsilon)
            time.sleep(1e-3)
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, 'a', positions)
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
