"""This module contains unit tests of the meshes module."""

import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from .. import meshes


def load_example_mesh():
    mesh_file = os.path.join(os.path.dirname(meshes.__file__),
                             'tests', 'example_mesh.ply')
    mesh = meshes.load_mesh(mesh_file)
    return mesh


def test_load_mesh():
    mesh = load_example_mesh()
    npt.assert_equal(mesh.shape, (10000, 3, 3))
    npt.assert_almost_equal(np.min(mesh), 0)
    return


def test_show_mesh():
    mesh = load_example_mesh()
    meshes.show_mesh(mesh, show=False)
    plt.close('all')
    return


def test__mesh_space_subdivision():
    mesh = load_example_mesh()
    sv_borders = meshes._mesh_space_subdivision(mesh, N=20)
    npt.assert_equal(sv_borders.shape, (3, 21))
    return


def test__interval_sv_overlap_1d():
    xs = np.arange(0, 11)
    inputs = [[0, 0], [10, 10], [2, -2], [7.2, 16]]
    outputs = [(0, 0), (10, 10), (0, 2), (7, 10)]
    for i, (x1, x2) in enumerate(inputs):
        ll, ul = meshes._interval_sv_overlap_1d(xs, x1, x2)
        npt.assert_equal((ll, ul), outputs[i])
    return


def test__subvoxel_to_triangle_mapping():
    mesh = load_example_mesh()
    sv_borders = meshes._mesh_space_subdivision(mesh, N=20)
    tri_indices, sv_mapping = meshes._subvoxel_to_triangle_mapping(
        mesh, sv_borders)
    desired_tri_indices = np.loadtxt(
        os.path.join(os.path.dirname(meshes.__file__), 'tests',
                     'desired_tri_indices.txt'))
    desired_sv_mapping = np.loadtxt(
        os.path.join(os.path.dirname(meshes.__file__), 'tests',
                     'desired_sv_mapping.txt'))
    npt.assert_equal(tri_indices, desired_tri_indices)
    npt.assert_equal(sv_mapping, desired_sv_mapping)
    return


def test__c_cross():
    np.random.seed(123)
    for i in range(10):
        A = np.random.random(3) - .5
        B = np.random.random(3) - .5
        C = meshes._c_cross(A, B)
        npt.assert_almost_equal(C, np.cross(A, B))


def test__c_dot():
    np.random.seed(123)
    for i in range(10):
        A = np.random.random(3) - .5
        B = np.random.random(3) - .5
        C = meshes._c_dot(A, B)
        npt.assert_almost_equal(C, np.dot(A, B))


def test__triangle_intersection_check():
    A = np.array([2.0, 0, 0])
    B = np.array([0, 2.0, 0])
    C = np.array([0.0, 0, 0])
    r0s = np.array([[0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [0.1, 0.1, 1.0],
                    [10, 10, 0]])
    steps = np.array([[0, 0, -1.0],
                      [0, 0, 1],
                      [0, 0, -.1],
                      [1.0, 1.0, 0],
                      [0, 0, 1.0]])
    desired = np.array([1, -1, 1, np.nan, np.nan])
    for i, (r0, step) in enumerate(zip(r0s, steps)):
        d = meshes._triangle_intersection_check(A, B, C, r0, step)
        npt.assert_almost_equal(d, desired[i])


def test__fill_mesh():
    mesh_file = os.path.join(os.path.dirname(meshes.__file__),
                             'tests', 'sphere_mesh.ply')
    mesh = meshes.load_mesh(mesh_file)
    sv_borders = meshes._mesh_space_subdivision(mesh, N=20)
    tri_indices, sv_mapping = meshes._subvoxel_to_triangle_mapping(
        mesh, sv_borders)
    n_s = int(1e5)
    points = meshes._fill_mesh(n_s, mesh, sv_borders, tri_indices,
                               sv_mapping, intra=True, extra=False)
    npt.assert_equal(points.shape, (n_s, 3))
    npt.assert_equal(np.max(np.linalg.norm(points - np.array([.1, .1, .1]),
                                           axis=1)) < .1, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0.1, 3)
    points = meshes._fill_mesh(n_s, mesh, sv_borders, tri_indices,
                               sv_mapping, intra=False, extra=True)
    npt.assert_equal(points.shape, (n_s, 3))
    npt.assert_equal(np.max(np.linalg.norm(points - np.array([.1, .1, .1]),
                                           axis=1)) > .1, True)
    npt.assert_almost_equal(np.mean(points, axis=0), 0.1, 3)
    return


def test__AABB_to_mesh():
    A = np.array([0, 0, 0])
    B = np.array([np.pi, np.pi / 2, np.pi * 2])
    mesh = meshes._AABB_to_mesh(A, B)
    desired = np.array([[[0., 0., 0., ],
                         [3.14159265, 0., 0.],
                         [3.14159265, 1.57079633, 0.]],
                        [[0., 0., 0.],
                         [0., 1.57079633, 0.],
                         [3.14159265, 1.57079633, 0.]],
                        [[0., 0., 6.28318531],
                         [3.14159265, 0., 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 0., 6.28318531],
                         [0., 1.57079633, 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 0., 0.],
                         [3.14159265, 0., 0.],
                         [3.14159265, 0., 6.28318531]],
                        [[0., 0., 0.],
                         [0., 0., 6.28318531],
                         [3.14159265, 0., 6.28318531]],
                        [[0., 1.57079633, 0.],
                         [3.14159265, 1.57079633, 0.],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 1.57079633, 0.],
                         [0., 1.57079633, 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[0., 0., 0.],
                         [0., 1.57079633, 0.],
                         [0., 1.57079633, 6.28318531]],
                        [[0., 0., 0.],
                         [0., 0., 6.28318531],
                         [0., 1.57079633, 6.28318531]],
                        [[3.14159265, 0., 0.],
                         [3.14159265, 1.57079633, 0.],
                         [3.14159265, 1.57079633, 6.28318531]],
                        [[3.14159265, 0., 0.],
                         [3.14159265, 0., 6.28318531],
                         [3.14159265, 1.57079633, 6.28318531]]])
    npt.assert_almost_equal(mesh, desired)
    return
