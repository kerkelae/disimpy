"""This module contains tests of the substrates module."""

import os
import pickle
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
    # This is the last test to be written in this module!
    return


def test__cross_product():
    a = np.array([0.77530597, 0.01563718, 0.78352089])
    b = np.array([0.49369194, -0.14263695, 0.35284948])
    npt.assert_equal(substrates._cross_product(a, b), np.cross(a, b))
    return


def test__dot_product():
    a = np.array([0.77530597, 0.01563718, 0.78352089])
    b = np.array([0.49369194, -0.14263695, 0.35284948])
    npt.assert_equal(substrates._dot_product(a, b), np.dot(a, b))
    return


def test__triangle_box_overlap():
    triangle = np.array([[0.5, 0.7, 0.3],
                         [0.9, 0.5, 0.2],
                         [0.6, 0.9, 0.8]])
    box = np.array([[.1, .3, .1],
                    [.4, .7, .5]])
    npt.assert_equal(substrates._triangle_box_overlap(triangle, box), False)
    triangle = np.array([[0.4, 0.7, 0.2],
                         [0.9, 0.5, 0.2],
                         [0.6, 0.9, 0.2]])
    box = np.array([[.4, .4, .3],
                    [.5, .8, .6]])
    npt.assert_equal(substrates._triangle_box_overlap(triangle, box), False)
    triangle = np.array([[0.63149023, 0.44235872, 0.77212144],
                         [0.25125724, 0.00087658, 0.66026559],
                         [0.8319006, 0.52731735, 0.22859846]])
    box = np.array([[0.33109806, 0.16637023, 0.91545459],
                    [0.79806038, 0.83915475, 0.38118002]])
    npt.assert_equal(substrates._triangle_box_overlap(triangle, box), True)
    return


def manual_test__triangle_box_overlap():
    """Useful function for visually checking the performance of the triangle-
    box overlap function."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    triangle = np.random.random((3, 3)) - .25
    tri = Poly3DCollection(triangle)
    face_color = 'tab:blue'
    tri.set_facecolor('tab:green')
    ax.add_collection3d(tri)
    box = np.random.random((2, 3)) - .25
    vertices, faces = substrates._aabb_to_mesh(box[0], box[1])
    for idx in faces:
        tri = Poly3DCollection(vertices[idx], alpha=.5)
        tri.set_facecolor('tab:blue')
        ax.add_collection3d(tri)
    ax.set_title(substrates._triangle_box_overlap(triangle, box))
    plt.show()
    return triangle, box


def test__interval_sv_overlap_1d():
    xs = np.arange(11)
    npt.assert_equal(substrates._interval_sv_overlap(xs, 0, 0), (0, 1))
    npt.assert_equal(substrates._interval_sv_overlap(xs, 0, 1.5), (0, 2))
    npt.assert_equal(substrates._interval_sv_overlap(xs, 9.5, 1.5), (1, 10))
    npt.assert_equal(substrates._interval_sv_overlap(xs, -1.1, .5), (0, 1))
    npt.assert_equal(substrates._interval_sv_overlap(xs, 9.5, 11.5), (9, 10))
    return


def test__triangle_aabb():
    triangle = np.array([[0.5, 0.7, 0.3],
                         [0.9, 0.5, 0.2],
                         [0.6, 0.9, 0.8]])
    npt.assert_equal(
        substrates._triangle_aabb(triangle),
        np.vstack((np.min(triangle, axis=0), np.max(triangle, axis=0))))
    return


def test__box_subvoxel_overlap():
    xs = np.arange(6)
    ys = np.arange(11)
    zs = np.arange(21)
    box = np.array([[2.5, 5., 2.2],
                    [9.2, 9.5, 20]])
    subvoxels = np.array([[2, 5],
                          [5, 10],
                          [2, 20]])
    npt.assert_equal(
        substrates._box_subvoxel_overlap(box, xs, ys, zs), subvoxels)
    return


def test__mesh_space_subdivision():
    # Test automatically
    # AND
    # Check the functionality manually by reconstructing the full mesh by
    # looping over the subvoxels!
    return


def manual_test__mesh_space_subdivision(n_sv=np.array([10, 10, 10]),
                                        padding=np.zeros(3)):
    """Useful function for manually visualizing subvoxel division."""
    import meshio
    mesh = meshio.read('disimpy/tests/fibre.stl')
    faces = mesh.cells[0].data
    vertices = mesh.points    
    #mesh_path = os.path.join(
    #    os.path.dirname(substrates.__file__), 'tests',
    #    'cylinder_mesh_closed.pkl')
    #mesh_path = os.path.join(
    #os.path.dirname(substrates.__file__), 'tests', 'example_mesh.pkl')
    with open(mesh_path, 'rb') as f:
        example_mesh = pickle.load(f)
    faces = example_mesh['faces']
    vertices = example_mesh['vertices']
    vertices -= np.min(vertices, axis=0)
    vertices += padding
    voxel_size = np.max(vertices, axis=0) + padding
    xs, ys, zs, triangle_indices, subvoxel_indices = (
        substrates._mesh_space_subdivision(
            vertices, faces, voxel_size, n_sv))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x in range(n_sv[0]):
        for y in range(n_sv[1]):
            for z in range(n_sv[2]):
                counter = 0
                i = x * n_sv[1] * n_sv[2] + y * n_sv[2] + z
                face_color = np.random.random(3)
                for j in range(subvoxel_indices[i, 0], subvoxel_indices[i, 1]):
                    counter += 1
                    triangle = vertices[faces[triangle_indices[j]]]
                    tri = Poly3DCollection(triangle)
                    tri.set_facecolor(face_color)
                    ax.add_collection3d(tri)
    ax.set_xlim([0, voxel_size[0]])
    ax.set_ylim([0, voxel_size[1]])
    ax.set_zlim([0, voxel_size[2]])
    fig.tight_layout()
    plt.show()
    return


def test__aabb_to_mesh():
    box = np.array([[2.5, 5., 2.2],
                    [9.2, 9.5, 20.]])
    vertices = np.array([[2.5, 5., 2.2],
                         [9.2, 5., 2.2],
                         [9.2, 9.5, 2.2],
                         [9.2, 9.5, 20.],
                         [2.5, 9.5, 20.],
                         [2.5, 5., 20.],
                         [2.5, 9.5, 2.2],
                         [9.2, 5., 20.]])
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
    npt.assert_equal(
        substrates._aabb_to_mesh(box[0], box[1]), (vertices, faces))
    return
