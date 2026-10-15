"""Focused mesh-index checks through Numba's CUDA simulator."""

import itertools
import math
import os

import numpy as np
import pytest

from .. import _cuda, _mesh, simulations, substrates
from .._cuda import (
    create_xoroshiro128p_states,
    cuda,
    xoroshiro128p_uniform_float64,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_ENABLE_CUDASIM") != "1",
    reason="requires NUMBA_ENABLE_CUDASIM=1 before import",
)


@cuda.jit()
def _overlap_bounds(xs, intervals, bounds):
    thread_id = cuda.grid(1)
    if thread_id >= intervals.shape[0]:
        return
    x1 = intervals[thread_id, 0]
    x2 = intervals[thread_id, 1]
    bounds[thread_id, 0] = _cuda._ll_subvoxel_overlap(xs, x1, x2)
    bounds[thread_id, 1] = _cuda._ul_subvoxel_overlap(xs, x1, x2)
    bounds[thread_id, 2] = _cuda._ll_subvoxel_overlap_periodic(xs, x1, x2)
    bounds[thread_id, 3] = _cuda._ul_subvoxel_overlap_periodic(xs, x1, x2)


@cuda.jit()
def _advance_rng_states(states):
    index = cuda.grid(1)
    if index < states.shape[0]:
        xoroshiro128p_uniform_float64(states, index)


@cuda.jit()
def _classify_points(
    points,
    triangle_data,
    voxel_size,
    triangle_indices,
    subvoxel_indices,
    xs,
    ys,
    zs,
    n_sv,
    inside,
):
    thread_id = cuda.grid(1)
    if thread_id >= points.shape[0]:
        return
    inside[thread_id] = _cuda._cuda_mesh_inside(
        points[thread_id],
        triangle_data,
        voxel_size,
        triangle_indices,
        subvoxel_indices,
        xs,
        ys,
        zs,
        n_sv,
        triangle_data.shape[0],
    )


@cuda.jit()
def _indexed_intersections(
    starts,
    directions,
    lengths,
    triangle_data,
    xs,
    ys,
    zs,
    subvoxel_indices,
    triangle_indices,
    n_sv,
    distances,
    face_indices,
):
    thread_id = cuda.grid(1)
    if thread_id >= starts.shape[0]:
        return
    closest_triangle_index = cuda.local.array(1, np.int64)
    distances[thread_id] = _cuda._cuda_find_mesh_intersection(
        starts[thread_id],
        directions[thread_id],
        lengths[thread_id],
        triangle_data,
        xs,
        ys,
        zs,
        subvoxel_indices,
        triangle_indices,
        n_sv,
        closest_triangle_index,
    )
    face_indices[thread_id] = closest_triangle_index[0]


@cuda.jit()
def _mesh_collision_endpoints(
    starts, steps, normals, distances, epsilon, crosses, step_length, endpoints
):
    thread_id = cuda.grid(1)
    if thread_id >= starts.shape[0]:
        return
    r0 = cuda.local.array(3, np.float64)
    step = cuda.local.array(3, np.float64)
    normal = cuda.local.array(3, np.float64)
    for axis in range(3):
        r0[axis] = starts[thread_id, axis]
        step[axis] = steps[thread_id, axis]
        normal[axis] = normals[thread_id, axis]
    consumed = _cuda._cuda_resolve_mesh_collision(
        r0,
        step,
        distances[thread_id],
        normal,
        epsilon,
        crosses[thread_id],
    )
    remaining = step_length - consumed
    for axis in range(3):
        endpoints[thread_id, axis] = r0[axis] + step[axis] * remaining


@cuda.jit()
def _mesh_reflection_decisions(
    triangle_indices, n_membrane_faces, probabilities, draws, reflects
):
    thread_id = cuda.grid(1)
    if thread_id >= triangle_indices.shape[0]:
        return
    reflects[thread_id] = _cuda._cuda_mesh_hit_reflects(
        triangle_indices[thread_id],
        n_membrane_faces,
        probabilities[thread_id],
        draws[thread_id],
    )


def _ray_triangle_distance(triangle, start, direction):
    """CPU Moller-Trumbore reference matching the production inequalities."""
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    offset = start - triangle[0]
    p = np.cross(direction, edge_2)
    q = np.cross(offset, edge_1)
    determinant = np.dot(p, edge_1)
    if determinant == 0:
        return math.nan
    distance = np.dot(q, edge_2) / determinant
    u = np.dot(p, offset) / determinant
    v = np.dot(q, direction) / determinant
    if 0 <= u <= 1 and 0 <= v <= 1 and u + v <= 1:
        return distance
    return math.nan


def _brute_inside(points, vertices, faces):
    direction = np.array([1.0, 0.0, 0.0])
    result = []
    for point in points:
        intersections = 0
        for face in faces:
            distance = _ray_triangle_distance(vertices[face], point, direction)
            intersections += np.isfinite(distance) and distance > 0
        result.append(intersections % 2)
    return np.asarray(result)


def _brute_periodic_distances(starts, directions, lengths, vertices, faces, voxel_size):
    result = np.full(len(starts), np.inf)
    for ray_index, (start, direction, length) in enumerate(
        zip(starts, directions, lengths)
    ):
        end = start + direction * length
        copy_bounds = []
        for axis in range(3):
            lower = math.floor(min(start[axis], end[axis]) / voxel_size[axis]) - 1
            upper = math.ceil(max(start[axis], end[axis]) / voxel_size[axis]) + 1
            copy_bounds.append(range(lower, upper + 1))
        for copy_index in itertools.product(*copy_bounds):
            shift = np.asarray(copy_index) * voxel_size
            for face in faces:
                distance = _ray_triangle_distance(
                    vertices[face] + shift, start, direction
                )
                if (
                    np.isfinite(distance)
                    and distance > 0
                    and distance <= length
                    and distance < result[ray_index]
                ):
                    result[ray_index] = distance
    return result


def _duplicated_lookup(triangle_indices, subvoxel_indices):
    """Duplicate every CSR entry without changing its row membership."""
    duplicated = []
    duplicated_rows = np.empty_like(subvoxel_indices)
    cursor = 0
    for row_index, (start, end) in enumerate(subvoxel_indices):
        row = np.repeat(triangle_indices[start:end], 2)
        duplicated_rows[row_index] = (cursor, cursor + len(row))
        duplicated.extend(row)
        cursor += len(row)
    return np.asarray(duplicated, dtype=triangle_indices.dtype), duplicated_rows


def _box_substrate(size=1.0, n_sv=(4, 4, 4)):
    vertices, faces = _mesh._aabb_to_mesh(np.zeros(3), np.full(3, size, dtype=float))
    return substrates.mesh(
        vertices.astype(float),
        faces.astype(int),
        periodic=True,
        n_sv=np.asarray(n_sv, dtype=int),
        quiet=True,
    )


def test_packed_triangle_geometry_matches_source_mesh():
    substrate = _box_substrate()
    origins = substrate.vertices[substrate.faces[:, 0]]
    edge_1 = substrate.vertices[substrate.faces[:, 1]] - origins
    edge_2 = substrate.vertices[substrate.faces[:, 2]] - origins
    normals = np.cross(edge_1, edge_2)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    assert substrate.triangle_data.flags.c_contiguous
    assert substrate.triangle_data.dtype == np.float64
    np.testing.assert_array_equal(substrate.triangle_data[:, :3], origins)
    np.testing.assert_array_equal(substrate.triangle_data[:, 3:6], edge_1)
    np.testing.assert_array_equal(substrate.triangle_data[:, 6:9], edge_2)
    np.testing.assert_array_equal(substrate.triangle_data[:, 9:12], normals)


def test_mesh_collision_epsilon_consumes_path_length():
    starts = np.array([[-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    steps = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    endpoints = np.empty_like(starts)
    _mesh_collision_endpoints[2, 1](
        starts,
        steps,
        normals,
        np.full(2, 0.5),
        0.1,
        np.array([False, True]),
        1.0,
        endpoints,
    )

    np.testing.assert_allclose(
        endpoints,
        np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        rtol=0,
        atol=1e-15,
    )


def test_mesh_permeability_endpoints_are_exact():
    triangle_indices = np.array([0, 0, 0, 0, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    draws = np.array([0.0, 0.75, 0.0, 1.0, 0.0])
    reflects = np.empty(5, dtype=np.bool_)
    _mesh_reflection_decisions[5, 1](
        triangle_indices,
        1,
        probabilities,
        draws,
        reflects,
    )

    # p=0 always reflects, p=1 always crosses, and artificial voxel faces
    # always reflect independently of the supplied permeability and draw.
    np.testing.assert_array_equal(reflects, np.array([True, True, False, False, True]))


def test_subvoxel_bounds_include_exact_planes_and_periodic_seams():
    xs = np.array([0.0, 0.5, 1.0])
    intervals = np.array(
        [
            [0.5, 0.5],
            [0.25, 0.5],
            [0.5, 0.75],
            [1.0, 1.0],
            [0.0, 0.0],
            [-1.0, -1.0],
        ]
    )
    bounds = np.empty((len(intervals), 4), dtype=np.int64)
    _overlap_bounds[1, len(intervals)](xs, intervals, bounds)

    np.testing.assert_array_equal(
        bounds[:, :2],
        np.array([[1, 2], [0, 2], [1, 2], [1, 2], [0, 1], [0, 0]]),
    )
    np.testing.assert_array_equal(
        bounds[:, 2:],
        np.array([[1, 2], [0, 2], [1, 2], [1, 3], [-1, 1], [-3, -1]]),
    )


def test_nonperiodic_lookup_filter_remaps_both_csr_row_ends():
    triangle_indices = np.array([0, 8, 0, 1, 9, 2, 10, 3])
    subvoxel_indices = np.array([[0, 2], [2, 2], [2, 5], [5, 8]])

    filtered, rows = _mesh._filter_mesh_lookup(
        triangle_indices, subvoxel_indices, n_faces=4
    )

    np.testing.assert_array_equal(filtered, np.array([0, 0, 1, 2, 3]))
    np.testing.assert_array_equal(rows, np.array([[0, 1], [1, 1], [1, 3], [3, 5]]))


def test_internal_grid_plane_is_indexed_on_both_sides_and_hit():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10e-6, 0.0],
            [0.0, 10e-6, 10e-6],
            [0.0, 0.0, 10e-6],
        ]
    )
    substrate = substrates.mesh(
        vertices,
        np.array([[0, 1, 2], [0, 2, 3]]),
        periodic=False,
        padding=np.array([5e-6, 0.0, 0.0]),
        init_pos="uniform",
        n_sv=np.array([4, 4, 4]),
        quiet=True,
    )
    assert substrate.vertices[0, 0] == substrate.xs[2]
    slab_size = substrate.n_sv[1] * substrate.n_sv[2]
    for x_index in (1, 2):
        membrane_ids = []
        for start, end in substrate.subvoxel_indices[
            x_index * slab_size : (x_index + 1) * slab_size
        ]:
            membrane_ids.extend(
                substrate.triangle_indices[start:end][
                    substrate.triangle_indices[start:end] < substrate.n_membrane_faces
                ]
            )
        assert set(membrane_ids) == {0, 1}

    starts = np.array([[2e-6, 3e-6, 4e-6], [8e-6, 7e-6, 6e-6]])
    directions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    lengths = np.full(2, 4e-6)
    distances = np.empty(2)
    face_indices = np.empty(2, dtype=np.int64)
    _indexed_intersections[2, 1](
        starts,
        directions,
        lengths,
        substrate.triangle_data.copy(),
        substrate.xs.copy(),
        substrate.ys.copy(),
        substrate.zs.copy(),
        substrate.subvoxel_indices.copy(),
        substrate.triangle_indices.copy(),
        substrate.n_sv.copy(),
        distances,
        face_indices,
    )
    np.testing.assert_allclose(distances, np.full(2, 3e-6), rtol=0, atol=1e-18)
    assert np.all(face_indices < substrate.n_membrane_faces)


def test_nonperiodic_compartment_sampling_uses_filtered_lookup():
    vertices, faces = _mesh._aabb_to_mesh(np.zeros(3), np.ones(3))
    substrate = substrates.mesh(
        vertices.astype(float),
        faces.astype(int),
        periodic=False,
        padding=np.full(3, 0.5),
        n_sv=np.array([2, 2, 2]),
        quiet=True,
    )

    inside = simulations._fill_mesh(
        8, substrate, intra=True, seed=612, cuda_bs=4, max_batches=100
    )
    outside = simulations._fill_mesh(
        8, substrate, intra=False, seed=612, cuda_bs=4, max_batches=100
    )

    assert np.all(inside > 0.5)
    assert np.all(inside < 1.5)
    assert np.all(np.any((outside < 0.5) | (outside > 1.5), axis=1))


def test_nested_shell_extra_sampling_reaches_inner_cavity():
    outer_vertices, outer_faces = _mesh._aabb_to_mesh(np.full(3, 0.1), np.full(3, 0.9))
    inner_vertices, inner_faces = _mesh._aabb_to_mesh(np.full(3, 0.3), np.full(3, 0.7))
    substrate = substrates.mesh(
        np.vstack((outer_vertices, inner_vertices)),
        np.vstack((outer_faces, inner_faces + len(outer_vertices))),
        periodic=False,
        padding=np.full(3, 0.1),
        init_pos="extra",
        n_sv=np.array([5, 5, 5]),
        quiet=True,
    )
    assert _mesh._mesh_sampling_grid(substrate, intra=False) is None
    points = simulations._fill_mesh(80, substrate, intra=False, seed=613, cuda_bs=8)
    in_cavity = np.all((points > 0.3) & (points < 0.7), axis=1)
    outside_outer = np.any((points < 0.1) | (points > 0.9), axis=1)
    assert np.any(in_cavity)
    assert np.all(in_cavity | outside_outer)


def test_periodic_seam_faces_do_not_change_compartment_parity():
    first_vertices, first_faces = _mesh._aabb_to_mesh(np.zeros(3), np.full(3, 0.4e-6))
    second_vertices, second_faces = _mesh._aabb_to_mesh(
        np.full(3, 0.8e-6), np.full(3, 1e-6)
    )
    substrate = substrates.mesh(
        np.vstack((first_vertices, second_vertices)),
        np.vstack((first_faces, second_faces + 8)),
        periodic=True,
        init_pos="intra",
        n_sv=np.array([12, 12, 12]),
        quiet=True,
    )
    points = (
        np.array(
            [
                [0.2, 0.2, 0.2],
                [0.55, 0.2, 0.2],
                [0.95, 0.9, 0.9],
                [0.65, 0.9, 0.9],
                [0.5, 0.5, 0.5],
            ]
        )
        * 1e-6
    )
    stream = cuda.stream()
    device_mesh = simulations._upload_mesh(substrate, stream, classification=True)
    classified = cuda.device_array(len(points), dtype=np.int32, stream=stream)
    _classify_points[1, len(points), stream](
        cuda.to_device(points, stream=stream),
        device_mesh["triangle_data"],
        device_mesh["voxel_size"],
        device_mesh["classification_triangle_indices"],
        device_mesh["classification_subvoxel_indices"],
        device_mesh["xs"],
        device_mesh["ys"],
        device_mesh["zs"],
        device_mesh["n_sv"],
        classified,
    )
    np.testing.assert_array_equal(
        classified.copy_to_host(stream=stream), [1, 0, 1, 0, 0]
    )

    sampled = simulations._fill_mesh(64, substrate, intra=True, seed=123, cuda_bs=8)
    first = np.all((sampled >= 0) & (sampled < 0.4e-6), axis=1)
    second = np.all((sampled >= 0.8e-6) & (sampled <= 1e-6), axis=1)
    assert np.all(first | second)


def test_mesh_compaction_handles_non_power_of_two_and_partial_blocks():
    vertices, faces = _mesh._aabb_to_mesh(np.zeros(3), np.ones(3))
    substrate = substrates.mesh(
        vertices.astype(float),
        faces.astype(int),
        periodic=False,
        padding=np.full(3, 0.5),
        n_sv=np.array([2, 2, 2]),
        quiet=True,
    )

    reference = simulations._fill_mesh(
        13, substrate, intra=True, seed=793, cuda_bs=1, max_batches=100
    )
    for block_size in (7, 96):
        sampled = simulations._fill_mesh(
            13,
            substrate,
            intra=True,
            seed=793,
            cuda_bs=block_size,
            max_batches=100,
        )
        np.testing.assert_array_equal(sampled, reference)


def test_sparse_mesh_initialization_grows_candidate_batches():
    # The interior occupies only 2.3% of the voxel. Fixed 16-candidate
    # batches run out before filling 16 positions with this seed.
    lower = np.full(3, 0.5)
    upper = np.full(3, 0.9)
    vertices, faces = _mesh._aabb_to_mesh(lower, upper)
    substrate = substrates.mesh(
        vertices.astype(float),
        faces.astype(int),
        periodic=False,
        padding=np.full(3, 0.5),
        n_sv=np.array([2, 2, 2]),
        quiet=True,
    )

    sampled = simulations._fill_mesh(
        16, substrate, intra=True, seed=221, cuda_bs=8, max_batches=32
    )
    assert sampled.shape == (16, 3)
    assert np.all(sampled > lower)
    assert np.all(sampled < upper)


def test_mesh_rng_growth_preserves_advanced_and_new_streams():
    stream = cuda.stream()
    original = create_xoroshiro128p_states(8, seed=174, stream=stream)
    _advance_rng_states[2, 4, stream](original)
    advanced = original.copy_to_host(stream=stream)
    enlarged = create_xoroshiro128p_states(32, seed=174, stream=stream)
    independent_new = enlarged.copy_to_host(stream=stream)
    _cuda._cuda_copy_rng_states[2, 4, stream](original, enlarged)
    grown = enlarged.copy_to_host(stream=stream)
    stream.synchronize()

    np.testing.assert_array_equal(grown[:8], advanced)
    np.testing.assert_array_equal(grown[8:], independent_new[8:])


def test_accelerated_inside_matches_all_triangles_on_boundaries_and_random_points():
    # A side length greater than one also verifies that the classifier traces
    # all the way to the voxel boundary rather than an arbitrary unit ray.
    substrate = _box_substrate(size=2.0)
    triangle_indices, subvoxel_indices = _duplicated_lookup(
        substrate.triangle_indices, substrate.subvoxel_indices
    )
    rng = np.random.default_rng(1947)
    points = np.vstack(
        (
            np.array(
                [
                    [0.25, 1.0, 0.75],
                    [1.5, 0.5, 1.0],
                    [-0.25, 1.0, 0.75],
                    [2.25, 1.0, 0.75],
                ]
            ),
            # Mesh rejection sampling only classifies candidates within the
            # simulated voxel. Explicit x-outside cases above still exercise
            # the ray endpoint handling without invoking periodic y/z copies.
            rng.uniform(0.01, 1.99, size=(96, 3)),
        )
    )
    inside = np.empty(len(points), dtype=np.int64)
    # One simulated thread per block avoids a Python 3.14 CUDASIM race in its
    # module-level ``cuda.local`` context while still executing device code.
    _classify_points[len(points), 1](
        points,
        substrate.triangle_data.copy(),
        substrate.voxel_size.copy(),
        triangle_indices,
        subvoxel_indices,
        substrate.xs.copy(),
        substrate.ys.copy(),
        substrate.zs.copy(),
        substrate.n_sv.copy(),
        inside,
    )

    np.testing.assert_array_equal(
        inside, _brute_inside(points, substrate.vertices, substrate.faces)
    )


def test_classifier_has_no_fixed_triangle_id_workspace():
    vertices = np.array([[0.75, 0.0, 0.0], [0.75, 1.0, 0.0], [0.75, 0.0, 1.0]])
    triangle = _mesh._pack_triangle_geometry(vertices, np.array([[0, 1, 2]]))
    triangle_data = np.repeat(triangle, 1001, axis=0)
    triangle_indices = np.arange(1001, dtype=np.int64)
    subvoxel_indices = np.array([[0, 1001]], dtype=np.int64)
    point = np.array([[0.25, 0.2, 0.2]])
    inside = np.empty(1, dtype=np.int64)

    _classify_points[1, 1](
        point,
        triangle_data,
        np.ones(3),
        triangle_indices,
        subvoxel_indices,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.ones(3, dtype=np.int64),
        inside,
    )

    np.testing.assert_array_equal(inside, np.array([1]))


def test_mesh_simulation_uploads_geometry_once_for_sampling_and_steps(monkeypatch):
    substrate = _box_substrate(n_sv=(2, 2, 2))
    substrate.init_pos = "intra"
    upload = simulations._upload_mesh
    calls = 0

    def counted_upload(*args, **kwargs):
        nonlocal calls
        calls += 1
        return upload(*args, **kwargs)

    monkeypatch.setattr(simulations, "_upload_mesh", counted_upload)
    signals, positions = simulations.simulation(
        n_walkers=4,
        diffusivity=2e-9,
        gradient=np.zeros((1, 2, 3)),
        dt=1e-4,
        substrate=substrate,
        seed=421,
        final_pos=True,
        quiet=True,
        cuda_bs=4,
    )

    assert calls == 1
    assert np.all(np.isfinite(signals))
    assert np.all((positions >= 0) & (positions <= substrate.voxel_size))


@pytest.mark.parametrize("n_sv", [(1, 1, 1), (4, 4, 4), (8, 4, 4)])
def test_accelerated_periodic_hits_match_all_triangle_copies(n_sv):
    substrate = _box_substrate(n_sv=n_sv)
    triangle_indices, subvoxel_indices = _duplicated_lookup(
        substrate.triangle_indices, substrate.subvoxel_indices
    )
    rng = np.random.default_rng(8251)
    random_starts = rng.uniform(-1.5, 2.5, size=(96, 3))
    random_directions = rng.normal(size=(96, 3))
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
    starts = np.vstack(
        (
            np.array(
                [
                    [0.1, 0.5, 0.25],
                    [0.1, 0.3, 0.4],
                    [1.0, 1.0, 0.37],
                    [1.0, 1.0, 1.0],
                    [0.1, 1.0, 0.37],
                    [0.1, 1.0, 1.0],
                    [0.125, 0.125, 0.125],
                ]
            ),
            random_starts,
        )
    )
    directions = np.vstack(
        (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.4, 0.7, 0.2],
                    [1.0, 0.73, 0.41],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            random_directions,
        )
    )
    directions[:7] /= np.linalg.norm(directions[:7], axis=1)[:, None]
    lengths = np.concatenate(
        (np.array([1.5, 0.9, 2.3, 2.3, 1.5, 1.5, 2.0]), rng.uniform(0.1, 3.0, 96))
    )
    distances = np.empty(len(starts))
    face_indices = np.empty(len(starts), dtype=np.int64)

    _indexed_intersections[len(starts), 1](
        starts,
        directions,
        lengths,
        substrate.triangle_data.copy(),
        substrate.xs.copy(),
        substrate.ys.copy(),
        substrate.zs.copy(),
        subvoxel_indices,
        triangle_indices,
        substrate.n_sv.copy(),
        distances,
        face_indices,
    )
    expected = _brute_periodic_distances(
        starts,
        directions,
        lengths,
        substrate.vertices,
        substrate.faces,
        substrate.voxel_size,
    )

    np.testing.assert_allclose(distances, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(face_indices >= 0, np.isfinite(expected))
