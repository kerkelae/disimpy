"""This module contains code for creating substrate objects.

Substrate objects are used for storing information about the simulated
microstructure.
"""

import numpy as np

from . import _mesh

_PREPARED_MESH_ATTRIBUTES = frozenset(
    (
        "type",
        "vertices",
        "faces",
        "n_membrane_faces",
        "voxel_size",
        "periodic",
        "n_sv",
        "triangle_data",
        "xs",
        "ys",
        "zs",
        "triangle_indices",
        "subvoxel_indices",
        "_mesh_prepared",
    )
)


class _Substrate:
    """Class for storing information about the simulated microstructure."""

    def __setattr__(self, name, value):
        if self.__dict__.get("_mesh_prepared", False) and name in (
            _PREPARED_MESH_ATTRIBUTES
        ):
            raise AttributeError(f"prepared mesh attribute '{name}' is read-only")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if self.__dict__.get("_mesh_prepared", False) and name in (
            _PREPARED_MESH_ATTRIBUTES
        ):
            raise AttributeError(f"prepared mesh attribute '{name}' is read-only")
        object.__delattr__(self, name)

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
            # Geometry and lookup data form one prepared acceleration object.
            # Own every array so later changes to the constructor arguments
            # cannot make the packed triangles and CSR lookup disagree.
            self.vertices = np.array(kwargs["vertices"], copy=True, order="C")
            self.faces = np.array(kwargs["faces"], copy=True, order="C")
            # Faces added for a nonperiodic voxel are physical walls, not
            # permeable membrane triangles.
            self.n_membrane_faces = kwargs["n_membrane_faces"]
            self.voxel_size = np.array(kwargs["voxel_size"], copy=True, order="C")
            self.periodic = kwargs["periodic"]
            self.init_pos = (
                np.array(kwargs["init_pos"], copy=True, order="C")
                if isinstance(kwargs["init_pos"], np.ndarray)
                else kwargs["init_pos"]
            )
            self.n_sv = np.array(kwargs["n_sv"], copy=True, order="C")
            self.perm_prob = kwargs["perm_prob"]
            self.triangle_data = _mesh._pack_triangle_geometry(
                self.vertices, self.faces
            )
            if not kwargs["quiet"]:
                print("Dividing the mesh into subvoxels")
            (
                self.xs,
                self.ys,
                self.zs,
                self.triangle_indices,
                self.subvoxel_indices,
            ) = _mesh._mesh_space_subdivision(
                self.vertices, self.faces, self.voxel_size, self.n_sv
            )
            for prepared in (
                self.vertices,
                self.faces,
                self.voxel_size,
                self.n_sv,
                self.triangle_data,
                self.xs,
                self.ys,
                self.zs,
                self.triangle_indices,
                self.subvoxel_indices,
            ):
                prepared.flags.writeable = False
            if isinstance(self.init_pos, np.ndarray):
                self.init_pos.flags.writeable = False
            if not kwargs["quiet"]:
                print("Finished dividing the mesh into subvoxels")
            self._mesh_prepared = True


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
    if not isinstance(radius, float) or not np.isfinite(radius) or radius <= 0:
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
    if not isinstance(radius, float) or not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"Incorrect value ({radius}) for radius")
    if (
        not isinstance(orientation, np.ndarray)
        or orientation.shape != (3,)
        or not np.issubdtype(orientation.dtype, np.floating)
        or not np.all(np.isfinite(orientation))
    ):
        raise ValueError(f"Incorrect value ({orientation}) for orientation")
    # Scale first so normalization remains stable for very large finite vectors.
    scale = np.max(np.abs(orientation))
    if scale == 0:
        raise ValueError(f"Incorrect value ({orientation}) for orientation")
    orientation = orientation / scale
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
        or not np.all(np.isfinite(semiaxes))
        or np.any(semiaxes <= 0)
    ):
        raise ValueError(f"Incorrect value ({semiaxes}) for semiaxes")
    if (
        not isinstance(R, np.ndarray)
        or R.shape != (3, 3)
        or not np.issubdtype(R.dtype, np.floating)
        or not np.all(np.isfinite(R))
    ):
        raise ValueError(f"Incorrect value ({R}) for R")
    elif not np.isclose(np.linalg.det(R), 1) or not np.allclose(R.T @ R, np.eye(3)):
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
        Probability that a spin passes through a triangle on each encounter.
        This is a dimensionless, time-step-specific numerical parameter.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.

    Notes
    -----
    Mesh geometry, subdivision, and array-valued initial positions are copied
    into the returned object. Prepared arrays are read-only and their
    attributes cannot be replaced, keeping packed triangle data and subvoxel
    lookup coherent. The ``init_pos`` attribute can still be replaced between
    simulations.

    Before preparation, repeated indexed faces and coincident same-oriented
    faces are reduced to their first occurrence, zero-area faces are removed,
    and coincident opposite-oriented faces cancel. Coordinate coincidence for
    this normalization requires exact equality.

    Compartment sampling checks that every geometric edge has two incident
    faces after vertices equal within floating-point roundoff are merged. It
    does not detect self-intersections, overlapping faces, or nonmanifold
    vertex links; meshes with those features are unsupported.
    """
    if (
        not isinstance(vertices, np.ndarray)
        or vertices.ndim != 2
        or vertices.shape[1] != 3
        or vertices.shape[0] < 3
        or not np.issubdtype(vertices.dtype, np.floating)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError(f"Incorrect value ({vertices}) for vertices.")
    if (
        not isinstance(faces, np.ndarray)
        or faces.ndim != 2
        or faces.shape[1] != 3
        or faces.shape[0] == 0
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise ValueError(f"Incorrect value ({faces}) for faces.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces contains a vertex index outside vertices")
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError("faces contains a triangle with repeated vertex indices")
    faces = _mesh._normalize_mesh_faces(vertices, faces)
    if not isinstance(periodic, bool):
        raise ValueError(f"Incorrect value ({periodic}) for periodic")
    if (
        not isinstance(padding, np.ndarray)
        or padding.shape != (3,)
        or not np.issubdtype(padding.dtype, np.floating)
        or not np.all(np.isfinite(padding))
        or np.any(padding < 0)
    ):
        raise ValueError(f"Incorrect value ({padding}) for padding")
    if isinstance(init_pos, np.ndarray):
        if (
            init_pos.ndim != 2
            or init_pos.shape[1] != 3
            or not np.issubdtype(init_pos.dtype, np.floating)
            or not np.all(np.isfinite(init_pos))
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
        or np.any(n_sv <= 0)
    ):
        raise ValueError(f"Incorrect value ({n_sv}) for n_sv")
    if (
        (perm_prob != 0 and not isinstance(perm_prob, float))
        or not np.isfinite(perm_prob)
        or perm_prob < 0
        or perm_prob > 1
    ):
        raise ValueError(f"Incorrect value ({perm_prob}) for perm_prob.")
    if not isinstance(quiet, bool):
        raise ValueError(f"Incorrect value ({quiet}) for quiet")

    if isinstance(init_pos, str) and init_pos in ("intra", "extra"):
        _mesh._validate_closed_surface(vertices, faces, init_pos)
    if not quiet:
        print("Aligning the corner of the simulated voxel with the origin")
    vertices, faces, voxel_size, n_membrane_faces, shift = _mesh._prepare_mesh_geometry(
        vertices, faces, padding, periodic
    )
    if not quiet:
        print(f"Moved the vertices by {shift}")
    if isinstance(init_pos, np.ndarray) and (
        np.any(init_pos < 0) or np.any(init_pos > voxel_size)
    ):
        raise ValueError("init_pos contains a point outside the simulated voxel")
    substrate = _Substrate(
        "mesh",
        vertices=vertices,
        faces=faces,
        n_membrane_faces=n_membrane_faces,
        voxel_size=voxel_size,
        n_sv=n_sv,
        periodic=periodic,
        init_pos=init_pos,
        quiet=quiet,
        perm_prob=perm_prob,
    )
    return substrate
