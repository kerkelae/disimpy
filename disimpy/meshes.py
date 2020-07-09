"""This module contains code for dealing with triangular meshes.

Triangular meshes are represented by numpy.ndarray instances of shape (number of
triangles, 3, 3) where the second dimension indices correspond to different
triangle points and the third dimension indices correspond to the Cartesian
coordinates of triangle points.
"""

import math
import numba
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_mesh(mesh_file):
    """Return mesh array corresponding to a triangular mesh in a .ply file.

    Parameters
    ----------
    mesh_file : str
        Path to triangular mesh file in .ply format [1]_.

    Returns
    -------
    mesh : ndarray
        Mesh array of shape (n of triangles, 3, 3) where the second dimension
        indices correspond to different triangle points and the third dimension
        is the cartesian coordinates of triangle points.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/PLY_(file_format)
    """
    with open(mesh_file, 'r') as f:
        mesh_data = f.readlines()
    n_of_vertices = int([i for i in mesh_data if
                         i.startswith('element vertex ')][0].split()[-1])
    first_vertice_idx = mesh_data.index('end_header\n') + 1
    vertices = np.loadtxt(mesh_data[first_vertice_idx:first_vertice_idx +
                                    n_of_vertices])
    faces = np.loadtxt(mesh_data[first_vertice_idx + n_of_vertices::])[:, 1:4]
    mesh = np.zeros((faces.shape[0], 3, 3))
    for i in range(faces.shape[0]):
        mesh[i, :, :] = vertices[np.array(faces[i], dtype=int)]
    mesh = np.add(mesh, - np.min(np.min(mesh, 0), 0))
    return mesh


def show_mesh(mesh, show=True):
    """Show a visualization of a triangular mesh with random triangle colours.

    Parameters
    ----------
    mesh : ndarray
        Mesh array of shape (n of triangles, 3, 3) where the second dimension
        indices correspond to different triangle points and the third dimension
        is the cartesian coordinates of triangle.
    show : bool
        Boolean switch defining whether to render figure or not.

    Returns
    -------
    None
    """
    np.random.seed(123)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([np.min(np.min(mesh, 0), 0)[0],
                 np.max(np.max(mesh, 0), 0)[0]])
    ax.set_ylim([np.min(np.min(mesh, 0), 0)[1],
                 np.max(np.max(mesh, 0), 0)[1]])
    ax.set_zlim([np.min(np.min(mesh, 0), 0)[2],
                 np.max(np.max(mesh, 0), 0)[2]])
    for triangle in mesh:
        A = triangle[0, :]
        B = triangle[1, :]
        C = triangle[2, :]
        vtx = np.array([A, B, C])
        tri = Poly3DCollection([vtx], alpha=.5)
        face_color = np.random.random(3)
        tri.set_facecolor(face_color)
        ax.add_collection3d(tri)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.ticklabel_format(style='sci', scilimits=(0, 0))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return


def _mesh_space_subdivision(mesh, N=20):
    """Divide mesh volume into N**3 subvoxels.

    Parameters
    ----------
    mesh : ndarray
        Triangular mesh represented by an array of shape (number of triangles,
        3, 3) where the second dimension indices correspond to different
        triangle points and the third dimension representes the cartesian
        coordinates.
    N : int
        Number of subvoxels along each cartesian coordinate axis.

    Returns
    -------
    sv_borders : ndarray
        Array of shape (3, N + 1) representing the boundaries between the
        subvoxels along cartesian coordinate axes.
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
    """Return indices of subvoxels that overlap with interval [x1, x2].

    Parameters
    ----------
    xs : array_like
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
        for i in range(len(xs)):
            if xs[i] > xmin:
                ll = i - 1
                break
    if xmax >= xs[-1]:
        ul = len(xs) - 1
    elif xmax <= xs[0]:
        ul = 0
    else:
        ul = len(xs) - 1
        for i in range(len(xs)):
            if not (xs[i] < xmax):
                ul = i
                break
    return ll, ul


def _subvoxel_to_triangle_mapping(mesh, sv_borders):
    """Generate a mapping between subvoxels and relevant triangles.

    Parameters
    ----------
    mesh : ndarray
        Triangular mesh represented by an array of shape (number of triangles,
        3, 3) where the second dimension indices correspond to different
        triangle points and the third dimension representes the cartesian
        coordinates.
    sv_borders : ndarray
        Array of shape (3, N + 1) representing the boundaries between the
        subvoxels along cartesian coordinate axes.

    Returns
    -------
    tri_indices : array_like
        1D array containing the relevant triangle indices for all subvoxels.
    sv_mapping : ndarray
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
    A : array_like
        1D array of length 3 defining a point of the triangle.
    B : array_like
        1D array of length 3 defining a point of the triangle.
    C : array_like
        1D array of length 3 defining a point of the triangle.
    r0 : array_like
        1D array of length 3 defining the point from which the distance to the
        triangle is calculated.
    step : array_like
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
    step /= math.sqrt(_c_dot(step, step))
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
    mesh : ndarray
        Triangular mesh represented by an array of shape (number of triangles,
        3, 3) where the second dimension indices correspond to different
        triangle points and the third dimension representes the cartesian
        coordinates.
    sv_borders : ndarray
        Array of shape (3, N + 1) representing the boundaries between the
        subvoxels along cartesian coordinate axes.
    tri_indices : array_like
        1D array containing the relevant triangle indices for all subvoxels.
    sv_mapping : ndarray
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
    positions : ndarray
        Array of shape (n_s, 3) containing the calculated positions.
    """
    if (not intra) and (not extra):
        raise ValueError('At least one of the parameters intra and extra have '
                         + 'to be True')
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
                step = np.array([1.0, 0, 0]) * voxel_size
                x_ll, x_ul = _interval_sv_overlap_1d(sv_borders[0, :], r0[0],
                                                     (r0 + step)[0])
                y_ll, y_ul = _interval_sv_overlap_1d(sv_borders[1, :], r0[1],
                                                     (r0 + step)[1])
                z_ll, z_ul = _interval_sv_overlap_1d(sv_borders[2, :], r0[2],
                                                     (r0 + step)[2])
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
