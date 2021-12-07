"""This module contains code for creating substrate objects.

Substrate objects are used for storing information about the simulated
microstructure.
"""

import numpy as np
import numba


class _Substrate:
    """Class for storing information about the simulated microstructure."""

    def __init__(self, substrate_type, **kwargs):
        self.type = substrate_type
        if self.type == "sphere":
            self.radius = kwargs["radius"]
        elif self.type == "cylinder":
            self.radius = kwargs["radius"]
            self.orientation = kwargs["orientation"]
        elif self.type == "ellipsoid":
            self.semiaxes = kwargs["semiaxes"]
            self.R = kwargs["R"]
        elif self.type == "mesh":
            self.vertices = kwargs["vertices"]
            self.faces = kwargs["faces"]
            self.voxel_size = kwargs["voxel_size"]
            self.periodic = kwargs["periodic"]
            self.init_pos = kwargs["init_pos"]
            self.n_sv = kwargs["n_sv"]
            if not kwargs["quiet"]:
                print("Dividing the mesh into subvoxels")
            (
                self.xs,
                self.ys,
                self.zs,
                self.triangle_indices,
                self.subvoxel_indices,
            ) = _mesh_space_subdivision(
                self.vertices, self.faces, self.voxel_size, self.n_sv
            )
            if not kwargs["quiet"]:
                print("Finished dividing the mesh into subvoxels")


def free():
    """Return a substrate object for simulating free diffusion.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
    """
    substrate = _Substrate("free")
    return substrate


def sphere(radius):
    """Return a substrate object for simulating diffusion in a sphere.

    Parameters
    ----------
    radius : float
        Radius of the sphere.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
    """
    if not isinstance(radius, float) or radius <= 0:
        raise ValueError(f"Incorrect value ({radius}) for radius")
    substrate = _Substrate("sphere", radius=radius)
    return substrate


def cylinder(radius, orientation):
    """Return a substrate object for simulating diffusion in an infinite
    cylinder.

    Parameters
    ----------
    radius : float
        Radius of the cylinder.
    orientation : numpy.ndarray
        Floating-point arrray with shape (3,) defining the orientation of the
        cylinder.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    if not isinstance(radius, float) or radius <= 0:
        raise ValueError(f"Incorrect value ({radius}) for radius")
    if (
        not isinstance(orientation, np.ndarray)
        or orientation.shape != (3,)
        or not np.issubdtype(orientation.dtype, np.floating)
    ):
        raise ValueError(f"Incorrect value ({orientation}) for orientation")
    orientation = orientation / np.linalg.norm(orientation)
    substrate = _Substrate("cylinder", radius=radius, orientation=orientation)
    return substrate


def ellipsoid(semiaxes, R=np.eye(3)):
    """Return a substrate object for simulating diffusion in an ellipsoid.

    Parameters
    ----------
    semiaxes : numpy.ndarray
        Floating-point array with shape (3,) containing the semiaxes of the
        axis-aligned ellipsoid.
    R : numpy.ndarray, optional
        Floating-point array with shape (3, 3) containing the rotation matrix
        that is applied to the axis-aligned ellipsoid before the simulation.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    if (
        not isinstance(semiaxes, np.ndarray)
        or semiaxes.shape != (3,)
        or not np.issubdtype(semiaxes.dtype, np.floating)
    ):
        raise ValueError(f"Incorrect value ({semiaxes}) for semiaxes")
    if (
        not isinstance(R, np.ndarray)
        or R.shape != (3, 3)
        or not np.issubdtype(R.dtype, np.floating)
    ):
        raise ValueError(f"Incorrect value ({R}) for R")
    elif not np.isclose(np.linalg.det(R), 1) or not np.all(
        np.isclose(R.T, np.linalg.inv(R))
    ):
        raise ValueError(f"R ({R}) is not a valid rotation matrix")
    substrate = _Substrate("ellipsoid", semiaxes=semiaxes, R=R)
    return substrate


def mesh(
    vertices,
    faces,
    periodic,
    padding=np.zeros(3),
    init_pos="uniform",
    n_sv=np.array([50, 50, 50]),
    quiet=False,
):
    """Return a substrate object for simulating diffusion restricted by a
    triangular mesh. The size of the simulated voxel is equal to the axis-
    aligned bounding box of the triangles plus padding. The triangles are
    shifted so that the lower corner of the simulated voxel is at the origin.

    Parameters
    ----------
    vertices : numpy.ndarray
        Floating-point array with shape (number of vertices, 3) containing the
        vertices of the triangular mesh.
    faces : numpy ndarray
        Integer array with shape (number of triangles, 3) containing the vertex
        indices of the points of the triangles.
    periodic : bool, optional
        If True, periodic boundary conditions are used, i.e., the random
        walkers leaving the simulated voxel encounter infinitely repeating
        copies of the simulated voxel. If False, the boundaries of the
        simulated voxel are an impermeable surface.
    padding : np.ndarray, optional
        Floating-point array with shape (3,) defining how much empty space
        there is between the axis-aligned bounding box of the triangles and the
        boundaries of the simulated voxel on both sides along each axis.
    init_pos : numpy.ndarray or str, optional
        Floating-point array with shape (number of random walkers, 3) defining
        the initial position of the random walkers within the simulated voxel
        or one of the following strings: 'uniform', 'intra', or 'extra'. If
        'uniform', the initial positions are sampled from a uniform
        distribution over the simulated voxel. If 'intra', the initial
        positions are sampled from a uniform distribution inside the surface
        defined by the triangular mesh. If 'extra', the initial positions are
        sampled from a uniform distribution over the simulated voxel excluding
        the volume inside the surface defined by the triangular mesh. Note that
        the triangles must define a closed surface if 'intra' or 'extra' is
        used.
    n_sv : np.ndarray, optional
        Integer array with shape (3,) controlling the number of subvoxels into
        which the simulated voxel is divided to accelerate the collision check
        algorithm.
    quiet : bool, optional
        If True, updates on computation progress are not printed.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    if (
        not isinstance(vertices, np.ndarray)
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.issubdtype(vertices.dtype, np.floating)
    ):
        raise ValueError(f"Incorrect value ({vertices}) for vertices.")
    if (
        not isinstance(faces, np.ndarray)
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise ValueError(f"Incorrect value ({faces}) for faces.")
    if not isinstance(periodic, bool):
        raise ValueError(f"Incorrect value ({periodic}) for periodic")
    if (
        not isinstance(padding, np.ndarray)
        or padding.shape != (3,)
        or not np.issubdtype(padding.dtype, np.floating)
    ):
        raise ValueError(f"Incorrect value ({padding}) for padding")
    if isinstance(init_pos, np.ndarray):
        if (
            init_pos.ndim != 2
            or init_pos.shape[1] != 3
            or not np.issubdtype(init_pos.dtype, np.floating)
        ):
            raise ValueError(f"Incorrect value ({init_pos}) for init_pos")
    elif isinstance(init_pos, str):
        if not (init_pos == "uniform" or init_pos == "intra" or init_pos == "extra"):
            raise ValueError(f"Incorrect value ({init_pos}) for init_pos")
    else:
        raise ValueError(f"Incorrect value ({init_pos}) for init_pos")
    if (
        not isinstance(n_sv, np.ndarray)
        or n_sv.shape != (3,)
        or not np.issubdtype(n_sv.dtype, np.integer)
    ):
        raise ValueError(f"Incorrect value ({n_sv}) for n_sv")
    if not quiet:
        print("Aligning the corner of the simulated voxel with the origin")
    shift = -np.min(vertices, axis=0) + padding
    vertices = vertices + shift
    if not quiet:
        print(f"Moved the vertices by {shift}")
    voxel_size = np.max(vertices, axis=0) + padding
    if not periodic:  # Add the voxel boundaries to the triangles
        voxel_vertices, voxel_faces = _aabb_to_mesh(np.zeros(3), voxel_size)
        faces = np.vstack((faces, voxel_faces + len(vertices)))
        vertices = np.vstack((vertices, voxel_vertices))
    substrate = _Substrate(
        "mesh",
        vertices=vertices,
        faces=faces,
        voxel_size=voxel_size,
        n_sv=n_sv,
        periodic=periodic,
        init_pos=init_pos,
        quiet=quiet,
    )
    return substrate


@numba.jit()
def _cross_product(a, b):
    """Compiled function for calculating the cross product between two 1D
    arrays of length 3."""
    c = np.zeros(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


@numba.jit()
def _dot_product(a, b):
    """Compiled function for calculating the dot product between two 1D arrays
    of length 3."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@numba.jit()
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


@numba.jit()
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


@numba.jit()
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


@numba.jit()
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
    relevant_triangles = [[] for _ in range(np.product(n_sv))]

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
    for i, l in enumerate(relevant_triangles):
        triangle_indices += l
        subvoxel_indices[i, 0] = counter
        subvoxel_indices[i, 1] = counter + len(l)
        counter += len(l)
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
