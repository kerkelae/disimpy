"""CPU helpers for preparing triangular meshes."""

import numba
import numpy as np


@numba.jit(nopython=True)
def _cross_product(a, b):
    """Compiled function for calculating the cross product between two 1D
    arrays of length 3."""
    c = np.zeros(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


@numba.jit(nopython=True)
def _dot_product(a, b):
    """Compiled function for calculating the dot product between two 1D arrays
    of length 3."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@numba.jit(nopython=True)
def _triangle_box_overlap(triangle, box):
    """Check if a triangle overlaps with a box.

    Parameters
    ----------
    triangle : numpy.ndarray
        Array with shape (3, 3) where the first dimension corresponds to the
        points of the triangle.
    box : numpy.ndarray
        Array with shape (2, 3) so that box[0, :] and box[1, :] are the corners
        closest to and furthest from the origin.

    Returns
    -------
    bool

    Notes
    -----
    This function is based on an algorithm by Tomas Akenine-Möller presented
    in the manuscript "More Fast 3D Triangle-Box Overlap Testing".
    """

    # Move the box and triangle so that the box's centre is at the origin
    c = np.array([np.mean(box[:, i]) for i in range(3)])
    h = np.abs(box[1] - box[0]) / 2
    v = triangle - c
    e = np.eye(3)

    # Test the triangle AABB against the box
    box_aabb = np.array(
        [[np.min(v[:, i]) for i in range(3)], [np.max(v[:, i]) for i in range(3)]]
    )
    if np.all(box_aabb[0] > h) or np.all(box_aabb[1] < -h):
        return False

    # Test the plane in which the triangle is against the box
    f = np.array(
        [
            [v[1, 0] - v[0, 0], v[1, 1] - v[0, 1], v[1, 2] - v[0, 2]],
            [v[2, 0] - v[1, 0], v[2, 1] - v[1, 1], v[2, 2] - v[1, 2]],
            [v[0, 0] - v[2, 0], v[0, 1] - v[2, 1], v[0, 2] - v[2, 2]],
        ]
    )
    normal = _cross_product(f[0], f[1])
    corners = np.array(
        [
            [h[0], h[1], h[2]],
            [-h[0], -h[1], -h[2]],
            [-h[0], h[1], h[2]],
            [h[0], -h[1], -h[2]],
            [h[0], -h[1], h[2]],
            [-h[0], h[1], -h[2]],
            [h[0], h[1], -h[2]],
            [-h[0], -h[1], h[2]],
        ]
    )
    in_plane = False
    behind = np.zeros(8, dtype=numba.boolean)
    for i, point in enumerate(corners):
        dp = _dot_product(normal, v[0] - point)
        if dp == 0:
            in_plane = True
        else:
            behind[i] = _dot_product(normal, v[0] - point) > 0
    if not in_plane and (np.all(behind) or np.all(~behind)):
        return False

    # Test the triangle against the box
    p = np.zeros(3)
    for i in range(3):
        for j in range(3):
            a = _cross_product(e[i], f[j])
            r = _dot_product(h, np.abs(a))
            for k in range(3):
                p[k] = _dot_product(a, v[k])
            if np.min(p) > r or np.max(p) < -r:
                return False
    return True


@numba.jit(nopython=True)
def _interval_sv_overlap(xs, x1, x2):
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


@numba.jit(nopython=True)
def _triangle_aabb(triangle):
    """Calculate the axis-aligned bounding box of a triangle and return its
    closest and furthest points to the origin.

    Parameters
    ----------
    triangle : numpy.ndarray
        Array with shape (3, 3) where the first dimension corresponds to the
        points of the triangle.

    Returns
    -------
    numpy.ndarray
    """
    aabb = np.zeros((2, 3))
    for i in range(3):
        aabb[0, i] = np.min(triangle[:, i])
        aabb[1, i] = np.max(triangle[:, i])
    return aabb


@numba.jit(nopython=True)
def _box_subvoxel_overlap(box, xs, ys, zs):
    """Find the subvoxels which with a box overlaps and return the lowest and
    highest index of the subvoxels along each axis.

    Parameters
    ----------
    box : numpy.ndarray
        Array with shape (2, 3) so that box[0, :] and box[1, :] are the corners
        closest to and furthest from the origin.
    xs, ys, zs : numpy.ndarray
        Subvoxel boundaries along each axis.

    Returns
    -------
    numpy.ndarray
    """
    subvoxels = np.zeros((3, 2), dtype=np.int32)
    for i, a in enumerate([xs, ys, zs]):
        subvoxels[i] = _interval_sv_overlap(a, box[0, i], box[1, i])
    return subvoxels


def _mesh_space_subdivision(vertices, faces, voxel_size, n_sv):
    """Divide the voxel into subvoxels and return arrays for finding the
    triangles in given a subvoxel.

    Parameters
    ----------
    vertices : numpy.ndarray
        Floating-point array with shape (number of vertices, 3) containing the
        vertices of the triangular mesh.
    faces : numpy ndarray
        Integer array with shape (number of triangles, 3) containing the vertex
        indices of the points of the triangles.
    voxel_size : numpy.ndarray
        Floating-point array with shape (3,).
    n_sv : numpy.ndarray
        Integer array of size (3,) defining the number of subvoxels along each
        axis.

    Returns
    -------
    xs : numpy.ndarray
        Floating-point array with shape (n_sv[0],) containing the subvoxel
        boundaries along the x-axis.
    ys : numpy.ndarray
        Floating-point array with shape (n_sv[1],) containing the subvoxel
        boundaries along the y-axis.
    zs : numpy.ndarray
        Floating-point array with shape (n_sv[2],) containing the subvoxel
        boundaries along the z-axis.
    triangle_indices : numpy.ndarray
        One-dimensional integer array containing the triangle indices for all
        subvoxels.
    subvoxel_indices : numpy.ndarray
        Two-dimensional integer array that enables the triangles of a given
        subvoxel to be located in triangle_indices. The triangles in subvoxel i
        are the elements from subvoxel_indices[i, 0] to subvoxel_indices[i, 1].
    """

    # Define the subvoxel boundaries
    xs = np.linspace(0, voxel_size[0], n_sv[0] + 1)
    ys = np.linspace(0, voxel_size[1], n_sv[1] + 1)
    zs = np.linspace(0, voxel_size[2], n_sv[2] + 1)
    relevant_triangles = [[] for _ in range(np.prod(n_sv))]

    # Loop over the triangles
    for i, idx in enumerate(faces):
        triangle = vertices[idx]
        subvoxels = _box_subvoxel_overlap(_triangle_aabb(triangle), xs, ys, zs)
        for x in range(subvoxels[0, 0], subvoxels[0, 1]):
            for y in range(subvoxels[1, 0], subvoxels[1, 1]):
                for z in range(subvoxels[2, 0], subvoxels[2, 1]):
                    box = np.array(
                        [[xs[x], ys[y], zs[z]], [xs[x + 1], ys[y + 1], zs[z + 1]]]
                    )
                    if _triangle_box_overlap(triangle, box):
                        subvoxel = x * n_sv[1] * n_sv[2] + y * n_sv[2] + z
                        relevant_triangles[subvoxel].append(i)

    # Make the final arrays
    triangle_indices = []
    subvoxel_indices = np.zeros((len(relevant_triangles), 2))
    counter = 0
    for i, indices in enumerate(relevant_triangles):
        triangle_indices += indices
        subvoxel_indices[i, 0] = counter
        subvoxel_indices[i, 1] = counter + len(indices)
        counter += len(indices)
    triangle_indices = np.array(triangle_indices).astype(int)
    subvoxel_indices = subvoxel_indices.astype(int)
    return xs, ys, zs, triangle_indices, subvoxel_indices


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
