"""This module contains code for running diffusion-weighted MR simulations."""

import os
import sys
import time
import math
import warnings
import contextlib

import numpy as np
import numba
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
    xoroshiro128p_uniform_float64,
)

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
def _cuda_triangle_normal(triangle, normal):
    """Calculate the normal vector of a triangle.

    Parameters
    ----------
    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray
    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    v = cuda.local.array(3, numba.float64)
    k = cuda.local.array(3, numba.float64)
    for i in range(3):
        v[i] = triangle[0, i] - triangle[1, i]
        k[i] = triangle[0, i] - triangle[2, i]
    _cuda_cross_product(v, k, normal)
    _cuda_normalize_vector(normal)
    return


@cuda.jit(device=True)
def _cuda_get_triangle(i, vertices, faces, triangle):
    """Get the ith triangle from vertices and faces arrays.

    Parameters
    ----------
    i : int
    vertices : numba.cuda.cudadrv.devicearray.DeviceNDArray
    faces : numba.cuda.cudadrv.devicearray.DeviceNDArray
    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    for a in range(3):
        for b in range(3):
            triangle[a, b] = vertices[faces[i, a], b]
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
    """Calculate the distance from r0 to a circle centered at origin along
    step. r0 must be inside the circle.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    radius : float

    Returns
    -------
    float
    """
    A = step[0] ** 2 + step[1] ** 2
    B = 2 * (r0[0] * step[0] + r0[1] * step[1])
    C = r0[0] ** 2 + r0[1] ** 2 - radius ** 2
    d = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    return d


@cuda.jit(device=True)
def _cuda_line_sphere_intersection(r0, step, radius):
    """Calculate the distance from r0 to a sphere centered at origin along
    step. r0 must be inside the sphere.

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
    d = -dp + math.sqrt(dp ** 2 - (_cuda_dot_product(r0, r0) - radius ** 2))
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
    A = (step[0] / a) ** 2 + (step[1] / b) ** 2 + (step[2] / c) ** 2
    B = 2 * (
        a ** (-2) * step[0] * r0[0]
        + b ** (-2) * step[1] * r0[1]
        + c ** (-2) * step[2] * r0[2]
    )
    C = (r0[0] / a) ** 2 + (r0[1] / b) ** 2 + (r0[2] / c) ** 2 - 1
    d = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    return d


@cuda.jit(device=True)
def _cuda_ray_triangle_intersection_check(triangle, r0, step):
    """Check if a ray defined by r0 and step intersects with a triangle defined
    by A, B, and C. The output is the distance in units of step length from r0
    to intersection if intersection found, nan otherwise. This function is
    based on the Moller-Trumbore algorithm.

    Parameters
    ----------
    triangle : numba.cuda.cudadrv.devicearray.DeviceNDArray
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    float or nan
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
        step[i] = (v[i] - 2 * dp * normal[i] + intersection[i]) - intersection[i]
    _cuda_normalize_vector(step)
    for i in range(3):  # Move walker slightly away from the surface
        r0[i] = intersection[i] + epsilon * normal[i]
    return


@numba.jit(nopython=True)
def _set_seed(seed):
    """Set the pseudorandom number generator seed for compiled functions."""
    np.random.seed(seed)
    return


@numba.jit(nopython=True)
def _fill_circle(n, radius):
    """Sample n random points from a uniform distribution inside a circle."""
    i = 0
    filled = False
    points = np.zeros((n, 2))
    while not filled:
        p = (np.random.random(2) - 0.5) * 2 * radius
        if np.linalg.norm(p) < radius:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


@numba.jit(nopython=True)
def _fill_sphere(n, radius):
    """Sample n random points from a uniform distribution inside a sphere."""
    i = 0
    filled = False
    points = np.zeros((n, 3))
    while not filled:
        p = (np.random.random(3) - 0.5) * 2 * radius
        if np.linalg.norm(p) < radius:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


@numba.jit(nopython=True)
def _fill_ellipsoid(n, semiaxes):
    """Sample n random points from a uniform distribution inside an axis
    aligned ellipsoid with semi-axes a, b, and c."""
    filled = False
    points = np.zeros((n, 3))
    i = 0
    while not filled:
        p = (np.random.random(3) - 0.5) * 2 * semiaxes
        if np.sum((p / semiaxes) ** 2) < 1:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


def _initial_positions_cylinder(n_walkers, radius, R):
    """Calculate initial positions for spins in a cylinder whose orientation is
    defined by R which defines the rotation from cylinder frame to lab
    frame."""
    positions = np.zeros((n_walkers, 3))
    positions[:, 1:3] = _fill_circle(n_walkers, radius)
    positions = np.matmul(R, positions.T).T
    return positions


def _initial_positions_ellipsoid(n_walkers, semiaxes, R):
    """Calculate initial positions for spins in an ellipsoid with semi-axes a,
    b, c whos whose orientation is defined by R which defines the rotation from
    ellipsoid frame to lab frame."""
    positions = _fill_ellipsoid(n_walkers, semiaxes)
    positions = np.matmul(R, positions.T).T
    return positions


@cuda.jit()
def _cuda_fill_mesh(
    points,
    rng_states,
    intra,
    vertices,
    faces,
    voxel_size,
    triangle_indices,
    subvoxel_indices,
    xs,
    ys,
    zs,
    n_sv,
):
    """Kernel function for efficiently sampling points from a uniform
    distribution inside or outside the surface defined by the triangular
    mesh."""

    thread_id = cuda.grid(1)
    if thread_id >= points.shape[0] or points[thread_id, 0] != math.inf:
        return

    point = cuda.local.array(3, numba.float64)
    for i in range(3):
        point[i] = xoroshiro128p_uniform_float64(rng_states, thread_id) * voxel_size[i]
    ray = cuda.local.array(3, numba.float64)
    ray[0] = 1.0
    ray[1] = 0.0
    ray[2] = 0.0

    # Find the subvoxels the ray intersects
    lls = cuda.local.array(3, numba.int64)
    uls = cuda.local.array(3, numba.int64)
    lls[0] = _ll_subvoxel_overlap(xs, point[0], point[0] + ray[0])
    lls[1] = _ll_subvoxel_overlap(ys, point[1], point[1] + ray[1])
    lls[2] = _ll_subvoxel_overlap(zs, point[2], point[2] + ray[2])
    uls[0] = _ul_subvoxel_overlap(xs, point[0], point[0] + ray[0])
    uls[1] = _ul_subvoxel_overlap(ys, point[1], point[1] + ray[1])
    uls[2] = _ul_subvoxel_overlap(zs, point[2], point[2] + ray[2])

    # Keep track of the number of intersections and the triangles. The max
    # number of intersections allowed is 1000. Increase this number for very
    # complex meshes.
    n_intersections = 0
    triangle = cuda.local.array((3, 3), numba.float64)
    triangles = cuda.local.array(1000, numba.int64)

    # Loop over the subvoxels
    for x in range(lls[0], uls[0]):
        for y in range(lls[1], uls[1]):
            for z in range(lls[2], uls[2]):
                sv = int(x * n_sv[1] * n_sv[2] + y * n_sv[2] + z)

                # Loop over the triangles
                for i in range(subvoxel_indices[sv, 0], subvoxel_indices[sv, 1]):

                    if n_intersections >= 1000:
                        return

                    _cuda_get_triangle(triangle_indices[i], vertices, faces, triangle)
                    d = _cuda_ray_triangle_intersection_check(triangle, point, ray)

                    if d > 0:
                        already_intersected = False
                        for j in triangles[0:n_intersections]:
                            if j == triangle_indices[i]:
                                already_intersected = True
                                break
                        if not already_intersected:
                            triangles[n_intersections] = triangle_indices[i]
                            n_intersections += 1

    if intra:
        if n_intersections % 2 == 1:  # Point is inside the surface
            for i in range(3):
                points[thread_id, i] = point[i]
    else:
        if n_intersections % 2 == 0:  # Point is outside the surface
            for i in range(3):
                points[thread_id, i] = point[i]
    return


def _fill_mesh(n_points, substrate, intra, seed, cuda_bs=128):
    """Sample points from a uniform distribution inside or outside the surface
    defined by a triangular mesh.

    Parameters
    ----------
    n_walkers : np.ndarray
    substrate : substrates._Substrate
    intra : bool
    seed : int
    cuda_bs : int, optional

    Returns
    -------
    points : np.ndarray
    """
    bs = cuda_bs
    gs = int(math.ceil(float(n_points) / bs))
    stream = cuda.stream()
    rng_states = create_xoroshiro128p_states(gs * bs, seed=seed, stream=stream)

    if substrate.periodic:
        d_vertices = cuda.to_device(substrate.vertices, stream=stream)
        d_faces = cuda.to_device(substrate.faces, stream=stream)
        d_subvoxel_indices = cuda.to_device(substrate.subvoxel_indices, stream=stream)
        d_triangle_indices = cuda.to_device(substrate.triangle_indices, stream=stream)
    else:  # Don't include the voxel boundaries
        vertices = np.copy(substrate.vertices[0:-8])
        faces = np.copy(substrate.faces[0:-12])
        triangle_indices = np.copy(substrate.triangle_indices)
        subvoxel_indices = np.copy(substrate.subvoxel_indices)
        to_delete = np.where(triangle_indices >= len(faces))[0]
        for n_deleted, i in enumerate(to_delete):
            for j in range(subvoxel_indices.shape[0]):
                if subvoxel_indices[j, 1] > i - n_deleted:
                    subvoxel_indices[j, :] -= 1
            triangle_indices = np.delete(triangle_indices, i - n_deleted)
        subvoxel_indices[subvoxel_indices < 0] = 0
        d_vertices = cuda.to_device(vertices, stream=stream)
        d_faces = cuda.to_device(faces, stream=stream)
        d_subvoxel_indices = cuda.to_device(subvoxel_indices, stream=stream)
        d_triangle_indices = cuda.to_device(triangle_indices, stream=stream)
    d_voxel_size = cuda.to_device(substrate.voxel_size, stream=stream)
    d_xs = cuda.to_device(substrate.xs, stream=stream)
    d_ys = cuda.to_device(substrate.ys, stream=stream)
    d_zs = cuda.to_device(substrate.zs, stream=stream)
    d_n_sv = cuda.to_device(substrate.n_sv, stream=stream)
    points_sampled = False
    points = np.array([])
    while not points_sampled:
        new_points = np.ones((n_points, 3)).astype(np.float64) * math.inf
        d_points = cuda.to_device(new_points, stream=stream)
        _cuda_fill_mesh[gs, bs, stream](
            d_points,
            rng_states,
            intra,
            d_vertices,
            d_faces,
            d_voxel_size,
            d_triangle_indices,
            d_subvoxel_indices,
            d_xs,
            d_ys,
            d_zs,
            d_n_sv,
        )
        stream.synchronize()
        new_points = d_points.copy_to_host(stream=stream)
        if len(points) == 0:
            points = new_points[~np.isinf(new_points)[:, 0]]
        else:
            points = np.vstack((points, new_points[~np.isinf(new_points)[:, 0]]))
        if points.shape[0] >= n_points:
            points_sampled = True
    return points[0:n_points]


def _aabb_to_mesh(a, b):
    """Return a triangular mesh that corresponds to an axis-aligned bounding
    box defined by points a and b."""
    vertices = np.array(
        [
            [a[0], a[1], a[2]],
            [b[0], a[1], a[2]],
            [b[0], b[1], a[2]],
            [b[0], b[1], b[2]],
            [a[0], b[1], b[2]],
            [a[0], a[1], b[2]],
            [a[0], b[1], a[2]],
            [b[0], a[1], b[2]],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
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
            [6, 4, 3],
        ]
    )
    return vertices, faces


@cuda.jit(device=True)
def _ll_subvoxel_overlap(xs, x1, x2):
    """For an interval [x1, x2], return the index of the lower limit of the
    overlapping subvoxels whose borders are defined by the elements of xs."""
    xmin = min(x1, x2)
    if xmin <= xs[0]:
        return 0
    elif xmin >= xs[-1]:
        ll = len(xs) - 1
        return ll
    else:
        for i, x in enumerate(xs):
            if x > xmin:
                ll = i - 1
                return ll
        ll = 0
        return ll


@cuda.jit(device=True)
def _ul_subvoxel_overlap(xs, x1, x2):
    """For an interval [x1, x2], return the index of the upper limit of the
    overlapping subvoxels whose borders are defined by the elements of xs."""
    xmax = max(x1, x2)
    if xmax >= xs[-1]:
        return len(xs) - 1
    elif xmax <= xs[0]:
        ul = 0
        return ul
    else:
        for i, x in enumerate(xs):
            if not x < xmax:
                ul = i
                return ul
        ul = len(xs) - 1
        return ul


@cuda.jit(device=True)
def _ll_subvoxel_overlap_periodic(xs, x1, x2):
    """For an interval [x1, x2], return the index of the lower limit of the
    overlapping subvoxels whose borders are defined by the elements of xs, and
    the division continues periodically."""
    xmin = min(x1, x2)
    voxel_size = abs(xs[-1] - xs[0])
    n = math.floor(xmin / voxel_size)  # How many voxel widths to shift
    xmin_shifted = xmin - n * voxel_size
    ll_shifted = _ll_subvoxel_overlap(xs, xmin_shifted, xmin_shifted)
    ll = ll_shifted + n * (len(xs) - 1)
    return ll


@cuda.jit(device=True)
def _ul_subvoxel_overlap_periodic(xs, x1, x2):
    """For an interval [x1, x2], return the index of the upper limit of the
    overlapping subvoxels whose borders are defined by the elements of xs, and
    the division continues periodically."""
    xmax = max(x1, x2)
    voxel_size = abs(xs[-1] - xs[0])
    n = math.floor(xmax / voxel_size)  # How many voxel widths to shift
    xmax_shifted = xmax - n * voxel_size
    ul_shifted = _ul_subvoxel_overlap(xs, xmax_shifted, xmax_shifted)
    ul = ul_shifted + n * (len(xs) - 1)
    return ul


@cuda.jit()
def _cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, t, step_l, dt):
    """Kernel function for free diffusion."""
    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.float64)
    _cuda_random_step(step, rng_states, thread_id)
    for i in range(3):
        positions[thread_id, i] = positions[thread_id, i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (
            GAMMA
            * dt
            * (
                (g_x[m, t] * positions[thread_id, 0])
                + (g_y[m, t] * positions[thread_id, 1])
                + (g_z[m, t] * positions[thread_id, 2])
            )
        )
    return


@cuda.jit()
def _cuda_step_sphere(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    radius,
    iter_exc,
    max_iter,
    epsilon,
):
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
        phases[m, thread_id] += (
            GAMMA
            * dt
            * (
                (g_x[m, t] * positions[thread_id, 0])
                + (g_y[m, t] * positions[thread_id, 1])
                + (g_z[m, t] * positions[thread_id, 2])
            )
        )
    return


@cuda.jit()
def _cuda_step_cylinder(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    radius,
    R,
    R_inv,
    iter_exc,
    max_iter,
    epsilon,
):
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
        phases[m, thread_id] += (
            GAMMA
            * dt
            * (
                (g_x[m, t] * positions[thread_id, 0])
                + (g_y[m, t] * positions[thread_id, 1])
                + (g_z[m, t] * positions[thread_id, 2])
            )
        )
    return


@cuda.jit()
def _cuda_step_ellipsoid(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    semiaxes,
    R,
    R_inv,
    iter_exc,
    max_iter,
    epsilon,
):
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
                normal[i] = -(r0[i] + d * step[i]) / semiaxes[i] ** 2
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
        phases[m, thread_id] += (
            GAMMA
            * dt
            * (
                (g_x[m, t] * positions[thread_id, 0])
                + (g_y[m, t] * positions[thread_id, 1])
                + (g_z[m, t] * positions[thread_id, 2])
            )
        )
    return


@cuda.jit()
def _cuda_step_mesh(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    vertices,
    faces,
    xs,
    ys,
    zs,
    subvoxel_indices,
    triangle_indices,
    iter_exc,
    max_iter,
    n_sv,
    epsilon,
):
    """Kernel function for diffusion restricted by a triangular mesh."""

    thread_id = cuda.grid(1)
    if thread_id >= positions.shape[0]:
        return

    # Allocate memory
    step = cuda.local.array(3, numba.float64)
    lls = cuda.local.array(3, numba.int64)
    uls = cuda.local.array(3, numba.int64)
    triangle = cuda.local.array((3, 3), numba.float64)
    normal = cuda.local.array(3, numba.float64)
    shifts = cuda.local.array(3, numba.float64)
    temp_r0 = cuda.local.array(3, numba.float64)

    # Get position and generate step
    r0 = positions[thread_id, :]
    _cuda_random_step(step, rng_states, thread_id)

    # Check for intersection, reflect step, and repeat until no intersection
    check_intersection = True
    iter_idx = 0
    while check_intersection and step_l > 0 and iter_idx < max_iter:
        iter_idx += 1
        min_d = math.inf

        # Find the relevant subvoxels for this step
        lls[0] = _ll_subvoxel_overlap_periodic(xs, r0[0], r0[0] + step[0] * step_l)
        lls[1] = _ll_subvoxel_overlap_periodic(ys, r0[1], r0[1] + step[1] * step_l)
        lls[2] = _ll_subvoxel_overlap_periodic(zs, r0[2], r0[2] + step[2] * step_l)
        uls[0] = _ul_subvoxel_overlap_periodic(xs, r0[0], r0[0] + step[0] * step_l)
        uls[1] = _ul_subvoxel_overlap_periodic(ys, r0[1], r0[1] + step[1] * step_l)
        uls[2] = _ul_subvoxel_overlap_periodic(zs, r0[2], r0[2] + step[2] * step_l)

        # Loop over subvoxels and fnd the closest triangle
        for x in range(lls[0], uls[0]):

            # Check if subvoxel is outside the simulated voxel
            if x < 0 or x > len(xs) - 2:
                shift_n = math.floor(x / (len(xs) - 1))
                x -= shift_n * (len(xs) - 1)
                shifts[0] = shift_n * xs[-1]
            else:
                shifts[0] = 0

            for y in range(lls[1], uls[1]):

                # Check if subvoxel is outside the simulated voxel
                if y < 0 or y > len(ys) - 2:
                    shift_n = math.floor(y / (len(ys) - 1))
                    y -= shift_n * (len(ys) - 1)
                    shifts[1] = shift_n * ys[-1]
                else:
                    shifts[1] = 0

                for z in range(lls[2], uls[2]):

                    # Check if subvoxel is outside the simulated voxel
                    if z < 0 or z > len(zs) - 2:
                        shift_n = math.floor(z / (len(zs) - 1))
                        z -= shift_n * (len(zs) - 1)
                        shifts[2] = shift_n * zs[-1]
                    else:
                        shifts[2] = 0

                    # Find the corresponding subvoxel in the simulated voxel
                    sv = int(x * n_sv[1] * n_sv[2] + y * n_sv[2] + z)

                    for i in range(3):  # Move walker to the simulated voxel
                        temp_r0[i] = r0[i] - shifts[i]

                    # Loop over the triangles in this subvoxel
                    for i in range(subvoxel_indices[sv, 0], subvoxel_indices[sv, 1]):
                        _cuda_get_triangle(
                            triangle_indices[i], vertices, faces, triangle
                        )
                        d = _cuda_ray_triangle_intersection_check(
                            triangle, temp_r0, step
                        )
                        if d > 0 and d < min_d:
                            closest_triangle_index = triangle_indices[i]
                            min_d = d

        # Check if step intersects with the closest triangle
        if min_d < step_l:
            _cuda_get_triangle(closest_triangle_index, vertices, faces, triangle)
            _cuda_triangle_normal(triangle, normal)
            _cuda_reflection(r0, step, min_d, normal, epsilon)
            step_l -= min_d
        else:
            check_intersection = False

    if iter_idx >= max_iter:
        iter_exc[thread_id] = True
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (
            GAMMA
            * dt
            * (
                (g_x[m, t] * positions[thread_id, 0])
                + (g_y[m, t] * positions[thread_id, 1])
                + (g_z[m, t] * positions[thread_id, 2])
            )
        )
    return


def add_noise_to_data(data, sigma, seed=None):
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
    if seed:
        np.random.seed(seed)
    noisy_data = np.abs(
        data
        + np.random.normal(size=data.shape, scale=sigma, loc=0)
        + 1j * np.random.normal(size=data.shape, scale=sigma, loc=0)
    )
    return noisy_data


def _write_traj(traj, mode, positions):
    """Write random walker trajectories to a file."""
    with open(traj, mode) as f:
        [f.write(str(i) + " ") for i in positions.ravel()]
        f.write("\n")
    return


def simulation(
    n_walkers,
    diffusivity,
    gradient,
    dt,
    substrate,
    seed=123,
    traj=None,
    final_pos=False,
    all_signals=False,
    quiet=False,
    cuda_bs=128,
    max_iter=int(1e3),
    epsilon=1e-13,
):
    """Simulate a diffusion-weighted MR experiment and generate signal. For a
    detailed tutorial, please see
    https://disimpy.readthedocs.io/en/latest/tutorial.html.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers.
    diffusivity : float
        Diffusivity in SI units (m^2/s).
    gradient : numpy.ndarray
        Floating-point array of shape (number of measurements, number of time
        points, 3). Array elements represent the gradient magnitude at a time
        point along an axis in SI units (T/m).
    dt : float
        Duration of a time step in the gradient array in SI units (s).
    substrate : disimpy.substrates._Substrate
        Substrate object containing information about the simulated
        microstructure.
    seed : int, optional
        Seed for pseudorandom number generation.
    traj : str, optional
        Path of a file in which to save the simulated random walker
        trajectories. The file can become very large! Every line represents a
        time point. Every line contains the positions as follows: walker_1_x
        walker_1_y walker_1_z walker_2_x walker_2_y walker_2_z…
    final_pos : bool, optional
        If True, the signal and the final positions of the random walkers are
        returned as a tuple.
    all_signals : bool, optional
        If True, the signal from each random walker is returned instead of the
        total signal.
    quiet : bool, optional
        If True, updates on the progress of the simulation are not printed.
    cuda_bs : int, optional
        The size of the one-dimensional CUDA thread block.
    max_iter : int, optional
        The maximum number of iterations allowed in the algorithm that checks
        if a random walker collides with a surface during a time step.
    epsilon : float, optional
        The amount by which a random walker is moved away from the surface
        after a collision to avoid placing it in the surface.

    Returns
    -------
    signal : numpy.ndarray
        Simulated signals.
    """

    # Confirm that Numba detects the GPU wihtout printing it
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        try:
            cuda.detect()
        except:
            raise Exception(
                "Numba was unable to detect a CUDA GPU. To run the simulation,"
                + " check that the requirements are met and CUDA installation"
                + " path is correctly set up: "
                + "https://numba.pydata.org/numba-doc/dev/cuda/overview.html"
            )

    # Validate input
    if not isinstance(n_walkers, int) or n_walkers <= 0:
        raise ValueError("Incorrect value (%s) for n_walkers" % n_walkers)
    if not isinstance(diffusivity, float) or diffusivity <= 0:
        raise ValueError("Incorrect value (%s) for diffusivity" % diffusivity)
    if (
        not isinstance(gradient, np.ndarray)
        or gradient.ndim != 3
        or gradient.shape[2] != 3
        or not np.issubdtype(gradient.dtype, np.floating)
    ):
        raise ValueError("Incorrect value (%s) for gradient" % gradient)
    if not isinstance(dt, float) or dt <= 0:
        raise ValueError("Incorrect value (%s) for dt" % dt)
    if not isinstance(substrate, substrates._Substrate):
        raise ValueError("Incorrect value (%s) for substrate" % substrate)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Incorrect value (%s) for seed" % seed)
    if traj:
        if not isinstance(traj, str):
            raise ValueError("Incorrect value (%s) for traj" % traj)
    if not isinstance(quiet, bool):
        raise ValueError("Incorrect value (%s) for quiet" % quiet)
    if not isinstance(cuda_bs, int) or cuda_bs <= 0:
        raise ValueError("Incorrect value (%s) for cuda_bs" % cuda_bs)
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("Incorrect value (%s) for max_iter" % max_iter)

    if not quiet:
        print("Starting simulation")
        if traj:
            print(
                "The trajectories file will be up to %s GB"
                % (gradient.shape[1] * n_walkers * 3 * 25 / 1e9)
            )

    # Set up CUDA stream
    bs = cuda_bs  # Threads per block
    gs = int(math.ceil(float(n_walkers) / bs))  # Blocks per grid
    stream = cuda.stream()

    # Set seed and create PRNG states
    np.random.seed(seed)
    _set_seed(seed)
    rng_states = create_xoroshiro128p_states(gs * bs, seed=seed, stream=stream)

    # Move arrays to the GPU
    d_g_x = cuda.to_device(np.ascontiguousarray(gradient[:, :, 0]), stream=stream)
    d_g_y = cuda.to_device(np.ascontiguousarray(gradient[:, :, 1]), stream=stream)
    d_g_z = cuda.to_device(np.ascontiguousarray(gradient[:, :, 2]), stream=stream)
    d_phases = cuda.to_device(np.zeros((gradient.shape[0], n_walkers)), stream=stream)
    d_iter_exc = cuda.to_device(np.zeros(n_walkers).astype(bool))

    # Calculate step length
    step_l = np.sqrt(6 * diffusivity * dt)

    if not quiet:
        print("Number of random walkers = %s" % n_walkers)
        print("Number of steps = %s" % gradient.shape[1])
        print("Step length = %s m" % step_l)
        print("Step duration = %s s" % dt)

    if substrate.type == "free":

        # Define initial positions
        positions = np.zeros((n_walkers, 3))
        if traj:
            _write_traj(traj, "w", positions)
        d_positions = cuda.to_device(positions, stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_free[gs, bs, stream](
                d_positions, d_g_x, d_g_y, d_g_z, d_phases, rng_states, t, step_l, dt,
            )
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, "a", positions)
            if not quiet:
                sys.stdout.write(f"\r{np.round((t / gradient.shape[1]) * 100, 1)}%")
                sys.stdout.flush()

    elif substrate.type == "cylinder":

        # Calculate rotation from lab frame to cylinder frame and back
        R = utils.vec2vec_rotmat(substrate.orientation, np.array([1.0, 0, 0]))
        R_inv = np.linalg.inv(R)
        d_R = cuda.to_device(R)
        d_R_inv = cuda.to_device(R_inv)

        # Calculate initial positions
        positions = _initial_positions_cylinder(n_walkers, substrate.radius, R_inv)
        if traj:
            _write_traj(traj, "w", positions)
        d_positions = cuda.to_device(positions, stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_cylinder[gs, bs, stream](
                d_positions,
                d_g_x,
                d_g_y,
                d_g_z,
                d_phases,
                rng_states,
                t,
                step_l,
                dt,
                substrate.radius,
                d_R,
                d_R_inv,
                d_iter_exc,
                max_iter,
                epsilon,
            )
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, "a", positions)
            if not quiet:
                sys.stdout.write(f"\r{np.round((t / gradient.shape[1]) * 100, 1)}%")
                sys.stdout.flush()

    elif substrate.type == "sphere":

        # Calculate initial positions
        positions = _fill_sphere(n_walkers, substrate.radius)
        if traj:
            _write_traj(traj, "w", positions)
        d_positions = cuda.to_device(positions, stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_sphere[gs, bs, stream](
                d_positions,
                d_g_x,
                d_g_y,
                d_g_z,
                d_phases,
                rng_states,
                t,
                step_l,
                dt,
                substrate.radius,
                d_iter_exc,
                max_iter,
                epsilon,
            )
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, "a", positions)
            if not quiet:
                sys.stdout.write(f"\r{np.round((t / gradient.shape[1]) * 100, 1)}%")
                sys.stdout.flush()

    elif substrate.type == "ellipsoid":

        d_semiaxes = cuda.to_device(substrate.semiaxes)

        # Calculate rotation from ellipsoid frame to lab frame and back
        R_inv = substrate.R
        d_R_inv = cuda.to_device(R_inv)
        d_R = cuda.to_device(np.linalg.inv(R_inv))

        # Calculate initial positions
        positions = _initial_positions_ellipsoid(n_walkers, substrate.semiaxes, R_inv)
        if traj:
            _write_traj(traj, "w", positions)
        d_positions = cuda.to_device(positions, stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_ellipsoid[gs, bs, stream](
                d_positions,
                d_g_x,
                d_g_y,
                d_g_z,
                d_phases,
                rng_states,
                t,
                step_l,
                dt,
                d_semiaxes,
                d_R,
                d_R_inv,
                d_iter_exc,
                max_iter,
                epsilon,
            )
            stream.synchronize()
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, "a", positions)
            if not quiet:
                sys.stdout.write(f"\r{np.round((t / gradient.shape[1]) * 100, 1)}%")
                sys.stdout.flush()

    elif substrate.type == "mesh":

        # Calculate initial positions
        if isinstance(substrate.init_pos, np.ndarray):
            if n_walkers != substrate.init_pos.shape[0]:
                raise ValueError(
                    "n_walkers must be equal to the number of initial positions"
                )
            positions = substrate.init_pos
        else:
            if not quiet:
                print("Calculating initial positions")
            if substrate.init_pos == "uniform":
                positions = np.random.random((n_walkers, 3)) * substrate.voxel_size
            elif substrate.init_pos == "intra":
                positions = _fill_mesh(n_walkers, substrate, True, seed)
            else:
                positions = _fill_mesh(n_walkers, substrate, False, seed)
            if not quiet:
                print("Finished calculating initial positions")
        if traj:
            _write_traj(traj, "w", positions)

        # Move arrays to the GPU
        d_vertices = cuda.to_device(substrate.vertices, stream=stream)
        d_faces = cuda.to_device(substrate.faces, stream=stream)
        d_xs = cuda.to_device(substrate.xs, stream=stream)
        d_ys = cuda.to_device(substrate.ys, stream=stream)
        d_zs = cuda.to_device(substrate.zs, stream=stream)
        d_triangle_indices = cuda.to_device(substrate.triangle_indices, stream=stream)
        d_subvoxel_indices = cuda.to_device(substrate.subvoxel_indices, stream=stream)
        d_n_sv = cuda.to_device(substrate.n_sv, stream=stream)
        d_positions = cuda.to_device(positions, stream=stream)

        # Run simulation
        for t in range(gradient.shape[1]):
            _cuda_step_mesh[gs, bs, stream](
                d_positions,
                d_g_x,
                d_g_y,
                d_g_z,
                d_phases,
                rng_states,
                t,
                step_l,
                dt,
                d_vertices,
                d_faces,
                d_xs,
                d_ys,
                d_zs,
                d_subvoxel_indices,
                d_triangle_indices,
                d_iter_exc,
                max_iter,
                d_n_sv,
                epsilon,
            )
            stream.synchronize()
            time.sleep(1e-2)
            if traj:
                positions = d_positions.copy_to_host(stream=stream)
                _write_traj(traj, "a", positions)
            if not quiet:
                sys.stdout.write(f"\r{np.round((t / gradient.shape[1]) * 100, 1)}%")
                sys.stdout.flush()

    else:
        raise ValueError("Incorrect value (%s) for substrate" % substrate)

    # Check if the intersection algorithm iteration limit was exceeded
    iter_exc = d_iter_exc.copy_to_host(stream=stream)
    if np.any(iter_exc):
        warnings.warn(
            "Maximum number of iterations was exceeded in the intersection "
            + "check algorithm for walkers %s" % np.where(iter_exc)[0]
        )

    # Calculate signal
    if all_signals:
        phases = d_phases.copy_to_host(stream=stream)
        phases[:, np.where(iter_exc)[0]] = np.nan
        signals = np.real(np.exp(1j * phases))
    else:
        phases = d_phases.copy_to_host(stream=stream)
        phases[:, np.where(iter_exc)[0]] = np.nan
        signals = np.real(np.nansum(np.exp(1j * phases), axis=1))
    if not quiet:
        sys.stdout.write("\rSimulation finished\n")
        sys.stdout.flush()
    if final_pos:
        positions = d_positions.copy_to_host(stream=stream)
        return signals, positions
    else:
        return signals
