"""This module contains tests of the substrates module."""

import os
import pickle

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from .. import _mesh, substrates


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


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -float("inf")])
def test_sphere_rejects_nonfinite_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        substrates.sphere(radius)


@pytest.mark.parametrize(
    "orientation",
    [
        np.zeros(3, dtype=float),
        np.array([np.nan, 0.0, 1.0]),
        np.array([np.inf, 0.0, 1.0]),
    ],
)
def test_cylinder_rejects_invalid_orientation(orientation):
    with pytest.raises(ValueError, match="orientation"):
        substrates.cylinder(5e-6, orientation)


@pytest.mark.parametrize(
    "semiaxes",
    [
        np.array([0.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([np.nan, 1.0, 1.0]),
        np.array([np.inf, 1.0, 1.0]),
    ],
)
def test_ellipsoid_rejects_invalid_semiaxes(semiaxes):
    with pytest.raises(ValueError, match="semiaxes"):
        substrates.ellipsoid(semiaxes)


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([1.0, 1.0, -1.0]),
        np.diag([1.0, 1.0, np.nan]),
        np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_ellipsoid_rejects_invalid_rotation(rotation):
    with pytest.raises(ValueError, match="R"):
        substrates.ellipsoid(np.ones(3), rotation)


def _tetrahedron():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-6, 0.0, 0.0],
            [0.0, 1e-6, 0.0],
            [0.0, 0.0, 1e-6],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {
                "vertices": np.array(
                    [[0.0, 0.0, np.nan], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                )
            },
            "vertices",
        ),
        ({"faces": np.array([[0, 1, 4]])}, "outside vertices"),
        ({"faces": np.array([[0, 1, -1]])}, "outside vertices"),
        ({"faces": np.array([[0, 1, 1]])}, "repeated vertex"),
        ({"padding": np.array([0.0, -1e-6, 0.0])}, "padding"),
        ({"padding": np.array([0.0, np.nan, 0.0])}, "padding"),
        ({"n_sv": np.array([1, 0, 1])}, "n_sv"),
        ({"init_pos": np.array([[np.nan, 0.0, 0.0]])}, "init_pos"),
        ({"init_pos": np.array([[2e-6, 0.0, 0.0]])}, "outside the simulated voxel"),
        ({"perm_prob": float("nan")}, "perm_prob"),
    ],
)
def test_mesh_rejects_invalid_geometry(change, message):
    vertices, faces = _tetrahedron()
    options = {
        "vertices": vertices,
        "faces": faces,
        "periodic": True,
        "n_sv": np.ones(3, dtype=int),
        "quiet": True,
    }
    options.update(change)
    with pytest.raises(ValueError, match=message):
        substrates.mesh(**options)


def test_mesh_rejects_zero_voxel_dimension_and_degenerate_triangle():
    with pytest.raises(ValueError, match="positive size along every axis"):
        substrates.mesh(
            np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0]]),
            np.array([[0, 1, 2]]),
            periodic=True,
            n_sv=np.ones(3, dtype=int),
            quiet=True,
        )

    with pytest.raises(ValueError, match="degenerate triangle"):
        substrates.mesh(
            np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [2e-6, 0.0, 0.0]]),
            np.array([[0, 1, 2]]),
            periodic=True,
            padding=np.ones(3),
            n_sv=np.ones(3, dtype=int),
            quiet=True,
        )


@pytest.mark.parametrize("init_pos", ["intra", "extra"])
def test_mesh_compartment_sampling_requires_closed_surface(init_pos):
    vertices, faces = _tetrahedron()
    with pytest.raises(ValueError, match="watertight mesh"):
        substrates.mesh(
            vertices,
            faces[:-1],
            periodic=True,
            init_pos=init_pos,
            n_sv=np.ones(3, dtype=int),
            quiet=True,
        )


def test_mesh_accepts_closed_surface_for_compartment_sampling():
    vertices, faces = _tetrahedron()
    substrate = substrates.mesh(
        vertices,
        faces,
        periodic=True,
        init_pos="intra",
        n_sv=np.ones(3, dtype=int),
        quiet=True,
    )
    npt.assert_array_equal(substrate.voxel_size, np.full(3, 1e-6))


def test_mesh_deduplicates_faces_stably_before_preparation():
    vertices, faces = _tetrahedron()
    faces_with_duplicates = np.vstack((faces, faces[2, ::-1], faces[0]))
    substrate = substrates.mesh(
        vertices,
        faces_with_duplicates,
        periodic=True,
        init_pos="intra",
        n_sv=np.ones(3, dtype=int),
        quiet=True,
    )

    npt.assert_array_equal(substrate.faces, faces)
    assert substrate.n_membrane_faces == len(faces)
    assert substrate.triangle_data.shape == (len(faces), 12)
    npt.assert_array_equal(substrate.triangle_indices, np.arange(len(faces)))


def test_mesh_normalizes_legacy_neuron_face_artifacts():
    mesh_path = os.path.join(
        os.path.dirname(substrates.__file__), "tests", "neuron-model.pkl"
    )
    with open(mesh_path, "rb") as f:
        neuron = pickle.load(f)

    substrate = substrates.mesh(
        neuron["vertices"],
        neuron["faces"],
        periodic=True,
        init_pos="intra",
        n_sv=np.ones(3, dtype=int),
        quiet=True,
    )

    # One indexed duplicate, 13 remaining zero-area faces, and one pair of
    # coincident opposite-facing triangles are absent from prepared geometry.
    assert substrate.n_membrane_faces == 29672
    assert len(substrate.faces) == 29672
    assert np.all(np.isfinite(substrate.triangle_data))
    assert len(np.unique(np.sort(substrate.faces, axis=1), axis=0)) == 29672


def test_mesh_accepts_closed_cylinder_with_welded_seam():
    mesh_path = os.path.join(
        os.path.dirname(substrates.__file__), "tests", "cylinder_mesh_closed.pkl"
    )
    with open(mesh_path, "rb") as f:
        cylinder = pickle.load(f)

    substrate = substrates.mesh(
        cylinder["vertices"],
        cylinder["faces"],
        periodic=True,
        init_pos="intra",
        n_sv=np.ones(3, dtype=int),
        quiet=True,
    )

    assert substrate.n_membrane_faces == len(cylinder["faces"])


def test_mesh():
    mesh_path = os.path.join(
        os.path.dirname(substrates.__file__),
        "tests",
        "sphere_mesh.pkl",
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
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces="f",
        periodic=True,
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
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=1,
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
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        perm_prob=-5.3,
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        perm_prob="t",
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        perm_prob=1,
    )
    npt.assert_raises(
        ValueError,
        substrates.mesh,
        vertices=vertices,
        faces=faces,
        periodic=True,
        perm_prob=1.00002,
    )
    substrate = substrates.mesh(vertices, faces, True)
    npt.assert_equal(substrate.type, "mesh")
    return


def test_mesh_owns_and_write_protects_prepared_arrays():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    n_sv = np.array([2, 2, 2], dtype=np.int64)
    init_pos = np.array([[0.25, 0.25, 0.25]])
    substrate = substrates.mesh(
        vertices,
        faces,
        periodic=True,
        padding=np.full(3, 0.1),
        init_pos=init_pos,
        n_sv=n_sv,
        quiet=True,
    )
    prepared_vertices = substrate.vertices.copy()
    prepared_faces = substrate.faces.copy()
    prepared_n_sv = substrate.n_sv.copy()
    prepared_init_pos = substrate.init_pos.copy()

    vertices[:] = 7
    faces[:] = 2
    n_sv[:] = 9
    init_pos[:] = 0.75
    np.testing.assert_array_equal(substrate.vertices, prepared_vertices)
    np.testing.assert_array_equal(substrate.faces, prepared_faces)
    np.testing.assert_array_equal(substrate.n_sv, prepared_n_sv)
    np.testing.assert_array_equal(substrate.init_pos, prepared_init_pos)

    prepared_names = (
        "vertices",
        "faces",
        "voxel_size",
        "n_sv",
        "triangle_data",
        "xs",
        "ys",
        "zs",
        "triangle_indices",
        "subvoxel_indices",
        "init_pos",
    )
    for name in prepared_names:
        prepared = getattr(substrate, name)
        assert not prepared.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            prepared.flat[0] = prepared.flat[0]

    # Prepared geometry and lookup arrays must remain a coherent unit.
    for name in (
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
    ):
        original = getattr(substrate, name)
        with pytest.raises(AttributeError, match="read-only"):
            setattr(substrate, name, original)
        assert getattr(substrate, name) is original
        with pytest.raises(AttributeError, match="read-only"):
            delattr(substrate, name)
        assert getattr(substrate, name) is original

    # Initial-position policy may still be changed between simulations.
    substrate.init_pos = "uniform"
    assert substrate.init_pos == "uniform"


def test__cross_product():
    np.random.seed(SEED)
    for _ in range(100):
        a = np.random.random(3) - 0.5
        b = np.random.random(3) - 0.5
        npt.assert_almost_equal(_mesh._cross_product(a, b), np.cross(a, b))
    return


def test__dot_product():
    np.random.seed(SEED)
    for _ in range(100):
        a = np.random.random(3) - 0.5
        b = np.random.random(3) - 0.5
        npt.assert_almost_equal(_mesh._dot_product(a, b), np.dot(a, b))
    return


def test__triangle_box_overlap():
    triangle = np.array([[0.5, 0.7, 0.3], [0.9, 0.5, 0.2], [0.6, 0.9, 0.8]])
    box = np.array([[0.1, 0.3, 0.1], [0.4, 0.7, 0.5]])
    npt.assert_equal(_mesh._triangle_box_overlap(triangle, box), False)
    triangle = np.array([[0.4, 0.7, 0.2], [0.9, 0.5, 0.2], [0.6, 0.9, 0.2]])
    box = np.array([[0.4, 0.4, 0.3], [0.5, 0.8, 0.6]])
    npt.assert_equal(_mesh._triangle_box_overlap(triangle, box), False)
    triangle = np.array(
        [
            [0.63149023, 0.44235872, 0.77212144],
            [0.25125724, 0.00087658, 0.66026559],
            [0.8319006, 0.52731735, 0.22859846],
        ]
    )
    box = np.array(
        [
            [0.33109806, 0.16637023, 0.91545459],
            [0.79806038, 0.83915475, 0.38118002],
        ]
    )
    npt.assert_equal(_mesh._triangle_box_overlap(triangle, box), True)
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
    vertices, faces = _mesh._aabb_to_mesh(box[0], box[1])
    for idx in faces:
        tri = Poly3DCollection(vertices[idx], alpha=0.5)
        tri.set_facecolor("tab:blue")
        ax.add_collection3d(tri)
    ax.set_title(_mesh._triangle_box_overlap(triangle, box))
    plt.show()
    return triangle, box


def test__interval_sv_overlap_1d():
    xs = np.arange(11)
    npt.assert_equal(_mesh._interval_sv_overlap(xs, 0, 0), (0, 1))
    npt.assert_equal(_mesh._interval_sv_overlap(xs, 0, 1.5), (0, 2))
    npt.assert_equal(_mesh._interval_sv_overlap(xs, 9.5, 1.5), (1, 10))
    npt.assert_equal(_mesh._interval_sv_overlap(xs, -1.1, 0.5), (0, 1))
    npt.assert_equal(_mesh._interval_sv_overlap(xs, 9.5, 11.5), (9, 10))
    return


def test__triangle_aabb():
    triangle = np.array([[0.5, 0.7, 0.3], [0.9, 0.5, 0.2], [0.6, 0.9, 0.8]])
    npt.assert_equal(
        _mesh._triangle_aabb(triangle),
        np.vstack((np.min(triangle, axis=0), np.max(triangle, axis=0))),
    )
    return


def test__box_subvoxel_overlap():
    xs = np.arange(6)
    ys = np.arange(11)
    zs = np.arange(21)
    box = np.array([[2.5, 5.0, 2.2], [9.2, 9.5, 20]])
    subvoxels = np.array([[2, 5], [5, 10], [2, 20]])
    npt.assert_equal(_mesh._box_subvoxel_overlap(box, xs, ys, zs), subvoxels)
    return


def test__mesh_space_subdivision():
    mesh_path = os.path.join(
        os.path.dirname(substrates.__file__),
        "tests",
        "sphere_mesh.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    voxel_size = np.max(vertices, axis=0)
    n_sv = np.array([2, 5, 10])
    xs, ys, zs, triangle_indices, subvoxel_indices = _mesh._mesh_space_subdivision(
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
        os.path.dirname(substrates.__file__),
        "tests",
        "cylinder_mesh_closed.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    vertices -= np.min(vertices, axis=0)
    vertices += padding
    voxel_size = np.max(vertices, axis=0) + padding
    # voxel_vertices, voxel_faces = _mesh._aabb_to_mesh(np.zeros(3), voxel_size)
    # faces = np.vstack((faces, voxel_faces + len(vertices)))
    # vertices = np.vstack((vertices, voxel_vertices))
    # vertices, faces = _mesh._aabb_to_mesh(np.zeros(3), voxel_size)
    (
        xs,
        ys,
        zs,
        triangle_indices,
        subvoxel_indices,
    ) = _mesh._mesh_space_subdivision(vertices, faces, voxel_size, n_sv)
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
    npt.assert_equal(_mesh._aabb_to_mesh(box[0], box[1]), (vertices, faces))
    return
