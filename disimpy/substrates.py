"""This module contains code for creating substrate objects.

Substrate objects are used for storing information about the simulated
microstructure.
"""

import numpy as np

from . import _mesh

_aabb_to_mesh = _mesh._aabb_to_mesh
_box_subvoxel_overlap = _mesh._box_subvoxel_overlap
_cross_product = _mesh._cross_product
_dot_product = _mesh._dot_product
_interval_sv_overlap = _mesh._interval_sv_overlap
_mesh_space_subdivision = _mesh._mesh_space_subdivision
_triangle_aabb = _mesh._triangle_aabb
_triangle_box_overlap = _mesh._triangle_box_overlap


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
            self.perm_prob = kwargs["perm_prob"]
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
    perm_prob=0,
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
    perm_prob : float, optional
        Probability that a spin passes through a triangle.

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
    if (
        (perm_prob != 0 and not isinstance(perm_prob, float))
        or perm_prob < 0
        or perm_prob > 1
    ):
        raise ValueError(f"Incorrect value ({perm_prob}) for perm_prob.")
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
        perm_prob=perm_prob,
    )
    return substrate
