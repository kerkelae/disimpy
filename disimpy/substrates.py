"""This module contains code for substrate objects that store information about
the simulated microstructure."""

import numpy as np


class _Substrate:
    """Class for storing information about the simulated microstructure."""

    def __init__(self, type, **kwargs):
        self.type = type
        if self.type == 'sphere':
            self.radius = kwargs['radius']
        elif self.type == 'cylinder':
            self.radius = kwargs['radius']
            self.orientation = kwargs['orientation']
        elif self.type == 'ellipsoid':
            self.semiaxes = kwargs['semiaxes']
            self.R = kwargs['R']
        elif self.type == 'mesh':
            self.triangles = kwargs['triangles']
            self.periodic = kwargs['periodic']
            self.init_pos = kwargs['init_pos']
            self.n_sv = kwargs['n_sv']


def free():
    """Return substrate object for simulating free diffusion.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    substrate = _Substrate('free')
    return substrate


def sphere(radius):
    """Return substrate object for simulating diffusion inside a sphere.

    Parameters
    ----------
    radius : float
        Radius of the simulated sphere.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    if not isinstance(radius, float) or radius <= 0:
        raise ValueError(
            'Incorrect value (%s) for radius.' % radius)
    substrate = _Substrate('sphere', radius=radius)
    return substrate


def cylinder(radius, orientation):
    """Return substrate object for simulating diffusion inside an infinite
    cylinder.

    Parameters
    ----------
    radius : float
        Radius of the simulated cylinder.
    orientation : numpy.ndarray
        Array of shape (3,) defining the orientation of the simulated cylinder.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    if not isinstance(radius, float) or radius <= 0:
        raise ValueError(
            'Incorrect value (%s) for radius.' % radius)
    if not isinstance(orientation, np.ndarray) or orientation.shape != (3,):
        raise ValueError(
            'Incorrect value (%s) for orientation.' % orientation)
    orientation = orientation.astype(float) / np.linalg.norm(orientation)
    substrate = _Substrate('cylinder', radius=radius, orientation=orientation)
    return substrate


def ellipsoid(semiaxes, R=np.eye(3)):
    """Return substrate object for simulating diffusion inside an ellipsoid.

    Parameters
    ----------
    semiaxes : numpy.ndarray
        Array of shape (3,) containing the semiaxes of an axis-aligned
        ellipsoid.
    R : numpy.ndarray, optional
        Rotation matrix of shape (3, 3) defining how the axis-aligned ellipsoid
        is rotated before the simulation.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    if not isinstance(semiaxes, np.ndarray) or semiaxes.shape != (3,):
        raise ValueError(
            'Incorrect value (%s) for semiaxes.' % semiaxes)
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError(
            'Incorrect value (%s) for R.' % R)
    semiaxes = semiaxes.astype(float)
    R = R.astype(float)
    substrate = _Substrate('ellipsoid', semiaxes=semiaxes, R=R)
    return substrate


def mesh(triangles, periodic=False, init_pos='uniform', n_sv=10):
    """Return substrate object for simulating diffusion restricted by a
    triangular mesh.

    Parameters
    ----------
    triangles : numpy.ndarray
        Array of shape (number of triangles, 3, 3) where the second dimension
        indices correspond to different triangle points and the third dimension
        indices correspond to the Cartesian coordinates of the points of the
        triangle. The triangles are shifted so that the corner of the axis-
        aligned bounding box of the triangles is at the origin.
    periodic : bool, optional
        If True, periodic boundary conditions are used, i.e., the random walkers
        can leave the axis-aligned bounding box of the set of triangles and
        encounter repeating copies of the mesh. If False, the axis-aligned
        bounding box of the triangles is considered an impermeable surface.
    init_pos : numpy.ndarray or str, optional
        An array of shape (number of random walkers, 3) defining the initial
        position of each walker within the axis-aligned bounding box of the
        triangles or one of the following strings: 'uniform', 'intra', 'extra'.
        If 'uniform', the initial positions are sampled from a uniform
        distribution over the axis-aligned bounding box of the triangles. If
        'intra', the initial positions are sampled from a uniform distribution
        inside the surface defined by the triangular mesh. If 'extra', the
        initial positions are sampled from a uniform distribution over the axis-
        aligned bounding box of the triangles excluding the volume inside the
        surface defined by the triangular mesh. Note that the triangles must
        define a closed surface if 'intra' or 'extra' are used. The initial
        positions are shifted like the triangles so that the corner of the axis-
        aligned bounding box of the triangles is at the origin.
    n_sv : tuple, optional
        Tuple of ints of length 3 defining the number of subvoxels along each
        axis into which the simulated voxel is divided to accelerate the
        algorithm that checks if a random walker step intersects with a
        triangle.

    Returns
    -------
    substrate : disimpy.substrates._Substrate
        Substrate object.
    """
    if (not isinstance(triangles, np.ndarray) or len(triangles.shape) != 3 or
            triangles.shape[1::] != (3, 3)):
        raise ValueError(
            'Incorrect value (%s) for triangles.' % triangles)
    if not isinstance(periodic, bool):
        raise ValueError(
            'Incorrect value (%s) for periodic.' % periodic)
    if init_pos != 'uniform' and init_pos != 'intra' and init_pos != 'extra':
        if (not isinstance(init_pos, np.ndarray) or len(init_pos) != 2 or
                init_pos.shape[1] != 3):
            raise ValueError(
                'Incorrect value (%s) for init_pos.' % init_pos)
    #if (not isinstance(n_sv, tuple) or len(n_sv) != 3 or
    #        np.any([not isinstance(i, int) for i in n_sv])):
    if not isinstance(n_sv, int) or n_sv < 1:
        raise ValueError(
            'Incorrect value (%s) for n_sv.' % n_sv)
    triangles = triangles - np.min(np.min(triangles, 0), 0)
    substrate = _Substrate(
        'mesh', triangles=triangles, periodic=periodic, init_pos=init_pos,
        n_sv=n_sv)
    return substrate
