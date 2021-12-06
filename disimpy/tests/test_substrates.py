"""This module contains tests of the substrates module."""

import os
import pickle

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from .. import substrates


SEED = 123


def test_free():
    substrate = substrates.free()
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.type, "free")
    return


def test_sphere():
    npt.assert_raises(ValueError, substrates.sphere, radius="r")
    npt.assert_raises(ValueError, substrates.sphere, radius=-5e-6)
    radius = 5e-6
    substrate = substrates.sphere(radius)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.radius, radius)
    npt.assert_equal(substrate.type, "sphere")
    return


def test_cylinder():
    orientation = np.array([1.0, 2, 0])
    npt.assert_raises(
        ValueError, substrates.cylinder, radius="r", orientation=orientation
    )
    npt.assert_raises(
        ValueError, substrates.cylinder, radius=-5e-6, orientation=orientation
    )
    radius = 5e-6
    npt.assert_raises(ValueError, substrates.cylinder, radius=radius, orientation="o")
    npt.assert_raises(
        ValueError, substrates.cylinder, radius=radius, orientation=np.arange(4)
    )
    npt.assert_raises(
        ValueError,
        substrates.cylinder,
        radius=radius,
        orientation=orientation.astype(int),
    )
    substrate = substrates.cylinder(radius, orientation)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.radius, radius)
    npt.assert_equal(substrate.orientation, orientation / np.linalg.norm(orientation))
    npt.assert_equal(substrate.type, "cylinder")
    return


def test_ellipsoid():
    npt.assert_raises(ValueError, substrates.ellipsoid, semiaxes="s")
    npt.assert_raises(ValueError, substrates.ellipsoid, semiaxes=np.arange(4))
    npt.assert_raises(
        ValueError, substrates.ellipsoid, semiaxes=np.arange(3).astype(int)
    )
    semiaxes = np.array([5e-6, 1e-6, 10e-6])
    npt.assert_raises(ValueError, substrates.ellipsoid, semiaxes=semiaxes, R="R")
    npt.assert_raises(ValueError, substrates.ellipsoid, semiaxes=semiaxes, R=np.eye(4))
    npt.assert_raises(
        ValueError, substrates.ellipsoid, semiaxes=semiaxes, R=np.eye(3).astype(int)
    )
    npt.assert_raises(
        ValueError, substrates.ellipsoid, semiaxes=semiaxes, R=np.zeros((3, 3))
    )
    substrate = substrates.ellipsoid(semiaxes)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.semiaxes, semiaxes)
    npt.assert_equal(substrate.R, np.eye(3))
    npt.assert_equal(substrate.type, "ellipsoid")
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).astype(float)
    substrate = substrates.ellipsoid(semiaxes, R)
    npt.assert_equal(isinstance(substrate, substrates._Substrate), True)
    npt.assert_equal(substrate.semiaxes, semiaxes)
    npt.assert_equal(substrate.R, R)
    npt.assert_equal(substrate.type, "ellipsoid")
    return


def test_mesh():
    mesh_path = os.path.join(
        os.path.dirname(substrates.__file__), "tests", "sphere_mesh.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    npt.assert_raises(
        ValueError, substrates.mesh, vertices="v", faces=faces, periodic=True
    )
    npt.assert_raises(
        ValueError, substrates.mesh, vertices=np.zeros(2), faces=faces, periodic=True
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=np.zeros((1, 4)),
        faces=faces,
        periodic=True,
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices.astype(int),
        faces=faces,
        periodic=True,
    )
    npt.assert_raises(
        ValueError, substrates.mesh, vertices=vertices, faces="f", periodic=True,
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=np.zeros(2).astype(int),
        periodic=True,
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=np.zeros((1, 4)).astype(int),
        periodic=True,
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces.astype(float),
        periodic=True,
    )
    npt.assert_raises(
        ValueError, substrates.mesh, vertices=vertices, faces=faces, periodic=1,
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        padding="p",
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        padding=np.zeros(2),
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        padding=np.ones(3).astype(int),
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        init_pos=np.zeros(1),
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        init_pos=np.zeros((1, 4)),
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        init_pos=np.zeros((1, 3)).astype(int),
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        init_pos="s",
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        n_sv="n",
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        n_sv=np.zeros((3, 3)),
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        n_sv=np.zeros((3)).astype(float),
    )
    substrate = substrates.mesh(vertices, faces, True)
    npt.assert_equal(substrate.type, "mesh")
    return


def test__cross_product():
    np.random.seed(SEED)
    for _ in range(100):
        a = np.random.random(3) - 0.5
        b = np.random.random(3) - 0.5
        npt.assert_almost_equal(substrates._cross_product(a, b), np.cross(a, b))
    return


def test__dot_product():
    np.random.seed(SEED)
    for _ in range(100):
        a = np.random.random(3) - 0.5
        b = np.random.random(3) - 0.5
        npt.assert_almost_equal(substrates._dot_product(a, b), np.dot(a, b))
    return


def test__triangle_box_overlap():
    triangle = np.array([[0.5, 0.7, 0.3], [0.9, 0.5, 0.2], [0.6, 0.9, 0.8]])
    box = np.array([[0.1, 0.3, 0.1], [0.4, 0.7, 0.5]])
    npt.assert_equal(substrates._triangle_box_overlap(triangle, box), False)
    triangle = np.array([[0.4, 0.7, 0.2], [0.9, 0.5, 0.2], [0.6, 0.9, 0.2]])
    box = np.array([[0.4, 0.4, 0.3], [0.5, 0.8, 0.6]])
    npt.assert_equal(substrates._triangle_box_overlap(triangle, box), False)
    triangle = np.array(
        [
            [0.63149023, 0.44235872, 0.77212144],
            [0.25125724, 0.00087658, 0.66026559],
            [0.8319006, 0.52731735, 0.22859846],
        ]
    )
    box = np.array(
        [[0.33109806, 0.16637023, 0.91545459], [0.79806038, 0.83915475, 0.38118002],]
    )
    npt.assert_equal(substrates._triangle_box_overlap(triangle, box), True)
    return


def manual_test__triangle_box_overlap():
    """Useful function for visually checking the performance of the triangle-
    box overlap function."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    triangle = np.random.random((3, 3)) - 0.25
    tri = Poly3DCollection(triangle)
    tri.set_facecolor("tab:green")
    ax.add_collection3d(tri)
    box = np.random.random((2, 3)) - 0.25
    vertices, faces = substrates._aabb_to_mesh(box[0], box[1])
    for idx in faces:
        tri = Poly3DCollection(vertices[idx], alpha=0.5)
        tri.set_facecolor("tab:blue")
        ax.add_collection3d(tri)
    ax.set_title(substrates._triangle_box_overlap(triangle, box))
    plt.show()
    return triangle, box


def test__interval_sv_overlap_1d():
    xs = np.arange(11)
    npt.assert_equal(substrates._interval_sv_overlap(xs, 0, 0), (0, 1))
    npt.assert_equal(substrates._interval_sv_overlap(xs, 0, 1.5), (0, 2))
    npt.assert_equal(substrates._interval_sv_overlap(xs, 9.5, 1.5), (1, 10))
    npt.assert_equal(substrates._interval_sv_overlap(xs, -1.1, 0.5), (0, 1))
    npt.assert_equal(substrates._interval_sv_overlap(xs, 9.5, 11.5), (9, 10))
    return


def test__triangle_aabb():
    triangle = np.array([[0.5, 0.7, 0.3], [0.9, 0.5, 0.2], [0.6, 0.9, 0.8]])
    npt.assert_equal(
        substrates._triangle_aabb(triangle),
        np.vstack((np.min(triangle, axis=0), np.max(triangle, axis=0))),
    )
    return


def test__box_subvoxel_overlap():
    xs = np.arange(6)
    ys = np.arange(11)
    zs = np.arange(21)
    box = np.array([[2.5, 5.0, 2.2], [9.2, 9.5, 20]])
    subvoxels = np.array([[2, 5], [5, 10], [2, 20]])
    npt.assert_equal(substrates._box_subvoxel_overlap(box, xs, ys, zs), subvoxels)
    return


def test__mesh_space_subdivision():
    mesh_path = os.path.join(
        os.path.dirname(substrates.__file__), "tests", "sphere_mesh.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    voxel_size = np.max(vertices, axis=0)
    n_sv = np.array([2, 5, 10])
    xs, ys, zs, triangle_indices, subvoxel_indices = substrates._mesh_space_subdivision(
        vertices, faces, voxel_size, n_sv
    )
    npt.assert_almost_equal(xs, np.linspace(0, voxel_size[0], n_sv[0] + 1))
    npt.assert_almost_equal(ys, np.linspace(0, voxel_size[1], n_sv[1] + 1))
    npt.assert_almost_equal(zs, np.linspace(0, voxel_size[2], n_sv[2] + 1))
    desired_triangle_indices = np.load(
        os.path.join(
            os.path.dirname(substrates.__file__),
            "tests",
            "desired_triangle_indices.npy",
        )
    )
    npt.assert_almost_equal(triangle_indices, desired_triangle_indices)
    desired_subvoxel_indices = np.load(
        os.path.join(
            os.path.dirname(substrates.__file__),
            "tests",
            "desired_subvoxel_indices.npy",
        )
    )
    npt.assert_almost_equal(subvoxel_indices, desired_subvoxel_indices)
    return


def manual_test__mesh_space_subdivision(n_sv=np.array([3, 3, 3]), padding=np.zeros(3)):
    """Useful function for manually visualizing subvoxel division."""
    mesh_path = os.path.join(
        os.path.dirname(substrates.__file__), "tests", "cylinder_mesh_closed.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    vertices -= np.min(vertices, axis=0)
    vertices += padding
    voxel_size = np.max(vertices, axis=0) + padding
    # voxel_vertices, voxel_faces = substrates._aabb_to_mesh(np.zeros(3), voxel_size)
    # faces = np.vstack((faces, voxel_faces + len(vertices)))
    # vertices = np.vstack((vertices, voxel_vertices))
    # vertices, faces = substrates._aabb_to_mesh(np.zeros(3), voxel_size)
    (
        xs,
        ys,
        zs,
        triangle_indices,
        subvoxel_indices,
    ) = substrates._mesh_space_subdivision(vertices, faces, voxel_size, n_sv)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for x in range(n_sv[0]):
        for y in range(n_sv[1]):
            for z in range(n_sv[2]):
                counter = 0
                i = x * n_sv[1] * n_sv[2] + y * n_sv[2] + z
                face_color = np.random.random(3)
                for j in range(subvoxel_indices[i, 0], subvoxel_indices[i, 1]):
                    counter += 1
                    triangle = vertices[faces[triangle_indices[j]]]
                    tri = Poly3DCollection(triangle, alpha=1)
                    tri.set_facecolor(face_color)
                    ax.add_collection3d(tri)
    ax.set_xlim([0, voxel_size[0]])
    ax.set_ylim([0, voxel_size[1]])
    ax.set_zlim([0, voxel_size[2]])
    fig.tight_layout()
    plt.show()
    return


def test__aabb_to_mesh():
    box = np.array([[2.5, 5.0, 2.2], [9.2, 9.5, 20.0]])
    vertices = np.array(
        [
            [2.5, 5.0, 2.2],
            [9.2, 5.0, 2.2],
            [9.2, 9.5, 2.2],
            [9.2, 9.5, 20.0],
            [2.5, 9.5, 20.0],
            [2.5, 5.0, 20.0],
            [2.5, 9.5, 2.2],
            [9.2, 5.0, 20.0],
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
    npt.assert_equal(substrates._aabb_to_mesh(box[0], box[1]), (vertices, faces))
    return
