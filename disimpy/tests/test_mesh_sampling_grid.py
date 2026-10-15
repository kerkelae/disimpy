"""Conservative sampling-grid checks independent of the collision grid."""

from types import SimpleNamespace

import numpy as np

from .._mesh import (
    _aabb_to_mesh,
    _mark_sampling_exterior,
    _mark_sampling_triangle_aabbs,
    _mesh_sampling_grid,
)


def _box(lower, upper):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _two_boxes(a_lower, a_upper, b_lower, b_upper, periodic=True):
    first_vertices, first_faces = _box(a_lower, a_upper)
    second_vertices, second_faces = _box(b_lower, b_upper)
    vertices = np.vstack((first_vertices, second_vertices))
    faces = np.vstack((first_faces, second_faces + 8))
    n_membrane_faces = len(faces)
    if not periodic:
        wall_vertices, wall_faces = _aabb_to_mesh(np.zeros(3), np.ones(3))
        faces = np.vstack((faces, wall_faces + len(vertices)))
        vertices = np.vstack((vertices, wall_vertices))
    return SimpleNamespace(
        vertices=vertices,
        faces=faces,
        n_membrane_faces=n_membrane_faces,
        voxel_size=np.ones(3),
        periodic=periodic,
    )


def _cell_of(point, shape, cell_size):
    xyz = np.minimum((np.asarray(point) / cell_size).astype(np.int32), shape - 1)
    return int(np.ravel_multi_index(xyz, shape))


def test_untouched_enclosed_cells_are_retained():
    vertices, faces = _box([0.2, 0.2, 0.2], [0.8, 0.8, 0.8])
    shape = np.array([20, 20, 20], dtype=np.int32)
    cell_size = np.ones(3) / shape
    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    _mark_sampling_triangle_aabbs(vertices, faces, len(faces), shape, cell_size, mask)
    interior = _cell_of([0.525, 0.525, 0.525], shape, cell_size)
    surface = _cell_of([0.225, 0.525, 0.525], shape, cell_size)
    assert mask[interior] == 0
    assert mask[surface] == 1
    _mark_sampling_exterior(shape, mask)
    assert mask[interior] == 0
    assert mask[surface] == 1
    assert mask[_cell_of([0.025, 0.025, 0.025], shape, cell_size)] == 2


def test_equal_volume_cells_yield_volume_weighted_uniform_samples():
    substrate = _two_boxes([0, 0, 0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8], [1, 1, 1])
    cell_ids, shape, cell_size = _mesh_sampling_grid(substrate)
    np.testing.assert_allclose(cell_size * shape, substrate.voxel_size)
    rng = np.random.default_rng(71)
    selected = rng.choice(cell_ids, size=300_000)
    xyz = np.column_stack(np.unravel_index(selected, shape))
    points = (xyz + rng.random((selected.size, 3))) * cell_size
    first = np.all((points >= 0) & (points < 0.4), axis=1)
    second = np.all((points >= 0.8) & (points < 1), axis=1)
    n_first = int(np.count_nonzero(first))
    n_second = int(np.count_nonzero(second))
    assert n_second > 500
    # The boxes have exact volume ratio 8:1. Rejection from equal-volume
    # proposal cells must preserve that ratio, including seam-adjacent cells.
    assert abs(n_first / n_second - 8) < 0.5
    np.testing.assert_allclose(points[first].mean(axis=0), 0.2, atol=0.01)
    np.testing.assert_allclose(points[second].mean(axis=0), 0.9, atol=0.01)


def test_periodic_seam_cover_matches_nonperiodic_membrane_cover():
    args = ([0, 0.4, 0.4], [0.1, 0.6, 0.6], [0.9, 0.4, 0.4], [1, 0.6, 0.6])
    periodic = _two_boxes(*args, periodic=True)
    nonperiodic = _two_boxes(*args, periodic=False)
    periodic_ids, shape, cell_size = _mesh_sampling_grid(periodic)
    nonperiodic_ids, other_shape, other_size = _mesh_sampling_grid(nonperiodic)
    np.testing.assert_array_equal(shape, other_shape)
    np.testing.assert_array_equal(cell_size, other_size)
    np.testing.assert_array_equal(periodic_ids, nonperiodic_ids)
    cover = set(periodic_ids.tolist())
    assert _cell_of([1e-6, 0.5, 0.5], shape, cell_size) in cover
    assert _cell_of([1 - 1e-6, 0.5, 0.5], shape, cell_size) in cover
    assert _cell_of([0.5, 0.5, 0.5], shape, cell_size) not in cover


def test_memory_cap_and_extra_proposal_policy():
    sparse = _two_boxes([0, 0.4, 0.4], [0.1, 0.6, 0.6], [0.9, 0.4, 0.4], [1, 0.6, 0.6])
    ids, shape, _ = _mesh_sampling_grid(sparse, max_bytes=32 * 500)
    assert np.prod(shape) <= 500
    assert ids.dtype == np.int32
    assert _mesh_sampling_grid(sparse, intra=False) is None

    # Flooding only from the voxel boundary excludes the inner cavity here,
    # despite that cavity being outside both closed shells by parity.
    nested = _two_boxes([0.05] * 3, [0.95] * 3, [0.25] * 3, [0.75] * 3)
    assert _mesh_sampling_grid(nested, intra=False) is None
