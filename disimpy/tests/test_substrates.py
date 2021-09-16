"""This module contains tests of the substrates module."""

import os
import numpy as np
import numpy.testing as npt

from .. import substrates


def test_free():
    substrate = substrates.free()
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    return


def test_sphere():
    npt.assert_raises(ValueError, substrates.sphere, radius='r')
    npt.assert_raises(ValueError, substrates.sphere, radius=-5e-6)
    radius = 5e-6
    substrate = substrates.sphere(radius)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.radius, radius)
    return


def test_cylinder():
    orientation = np.array([1., 2, 0])
    npt.assert_raises(
        ValueError, substrates.cylinder, radius='r', orientation=orientation)
    npt.assert_raises(
        ValueError, substrates.cylinder, radius=-5e-6, orientation=orientation)
    radius = 5e-6
    npt.assert_raises(
        ValueError, substrates.cylinder, radius=radius, orientation='o')
    npt.assert_raises(
        ValueError, substrates.cylinder, radius=radius,
        orientation=np.arange(4))
    npt.assert_raises(
        ValueError, substrates.cylinder, radius=radius,
        orientation=orientation.astype(int))
    substrate = substrates.cylinder(radius, orientation)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.radius, radius)
    npt.assert_equal(
        substrate.orientation, orientation / np.linalg.norm(orientation))
    return


def test_ellipsoid():
    npt.assert_raises(ValueError, substrates.ellipsoid, semiaxes='s')
    npt.assert_raises(ValueError, substrates.ellipsoid, semiaxes=np.arange(4))
    semiaxes = np.array([5e-6, 1e-6, 10e-6])
    npt.assert_raises(
        ValueError, substrates.ellipsoid, semiaxes=semiaxes, R='R')
    npt.assert_raises(
        ValueError, substrates.ellipsoid, semiaxes=semiaxes, R=np.eye(4))
    npt.assert_raises(
        ValueError, substrates.ellipsoid, semiaxes=semiaxes, R=np.zeros((3, 3)))
    substrate = substrates.ellipsoid(semiaxes)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.semiaxes, semiaxes)
    npt.assert_equal(substrate.R, np.eye(3))
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    substrate = substrates.ellipsoid(semiaxes, R)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.semiaxes, semiaxes)
    npt.assert_equal(substrate.R, R)
    return


def test_mesh():
    npt.assert_raises(ValueError, substrates.mesh, triangles='t')
    npt.assert_raises(ValueError, substrates.mesh, triangles=np.arange(3))
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=np.zeros((3, 3, 1)))
    mesh_file = os.path.join(
        os.path.dirname(substrates.__file__), 'tests', 'example_mesh.npy')
    triangles = np.load(mesh_file)
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles, padding='s')
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles, padding=np.arange(4))
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles, periodic='f')
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles, init_pos='i')
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles, init_pos=np.eye(4))
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles, n_sv='2')
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles, n_sv=0)
    npt.assert_raises(
        ValueError, substrates.mesh, triangles=triangles,
        init_pos=np.zeros((10, 3)) - 1)
    substrate = substrates.mesh(triangles)
    npt.assert_equal(substrate.type, 'mesh')
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_almost_equal(
        substrate.voxel_size, np.max(triangles, axis=(0, 1)))
    substrate = substrates.mesh(triangles, padding=np.ones(3) * 1e-6)
    npt.assert_almost_equal(
        substrate.voxel_size, np.max(triangles, axis=(0, 1)) + 2e-6)
    return
