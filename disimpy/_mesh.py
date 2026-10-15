"""CPU helpers for preparing triangular meshes."""

import numba
import numpy as np

_FLOAT64_EPS = np.finfo(np.float64).eps
_SAMPLING_GRID_MAX_BYTES = 128 * 1024 * 1024


def _normalize_mesh_faces(vertices, faces):
    """Remove nonphysical face artifacts while preserving input face order.

    Repeated indexed faces retain their first occurrence. Faces with zero
    geometric area are removed. If distinct vertex indices refer to identical
    coordinates, coincident faces retain their first occurrence regardless of
    orientation. Redundant folded flaps must be repaired explicitly in the
    input mesh: coordinate coincidence cannot distinguish them from physical
    membranes.
    """
    # Exact coordinate aliases occur in legacy STL conversions. Retain the
    # first triangle's vertex indices and order when deduplicating them.
    _, coordinate_indices = np.unique(vertices, axis=0, return_inverse=True)
    coordinate_faces = coordinate_indices[faces]
    repeated_coordinate = np.any(
        np.diff(np.sort(coordinate_faces, axis=1), axis=1) == 0, axis=1
    )
    vertices_64 = vertices.astype(np.float64, copy=False)
    edge_1 = vertices_64[faces[:, 1]] - vertices_64[faces[:, 0]]
    edge_2 = vertices_64[faces[:, 2]] - vertices_64[faces[:, 0]]
    collinear = np.all(np.cross(edge_1, edge_2) == 0, axis=1)
    keep = ~(repeated_coordinate | collinear)
    if not np.any(keep):
        raise ValueError("faces contains only degenerate triangles")
    faces = faces[keep]
    coordinate_faces = coordinate_faces[keep]

    canonical_coordinates = np.sort(coordinate_faces, axis=1)
    _, first_coordinate_face_indices = np.unique(
        canonical_coordinates, axis=0, return_index=True
    )
    faces = faces[np.sort(first_coordinate_face_indices)]

    if not len(faces):
        raise ValueError("faces contains no surface after normalization")
    return faces


def _validate_closed_surface(vertices, faces, compartment):
    """Require geometric edge incidence two for parity compartment sampling."""
    coordinate_scale = max(
        float(np.max(np.ptp(vertices, axis=0))),
        float(np.max(np.abs(vertices))),
    )
    weld_tolerance = 64 * _FLOAT64_EPS * coordinate_scale
    vertex_keys = np.rint(
        (vertices - np.min(vertices, axis=0)) / weld_tolerance
    ).astype(np.int64)
    _, welded_indices = np.unique(vertex_keys, axis=0, return_inverse=True)
    welded_faces = welded_indices[faces]
    edges = np.concatenate(
        (
            welded_faces[:, [0, 1]],
            welded_faces[:, [1, 2]],
            welded_faces[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(edge_counts != 2):
        raise ValueError(
            f"init_pos='{compartment}' requires a watertight mesh where every "
            "geometric edge belongs to exactly two faces"
        )


def _prepare_mesh_geometry(vertices, faces, padding, periodic):
    """Shift the mesh to voxel coordinates and append impermeable walls."""
    shift = -np.min(vertices, axis=0) + padding
    vertices = vertices + shift
    voxel_size = np.max(vertices, axis=0) + padding
    if not np.all(np.isfinite(voxel_size)) or np.any(voxel_size <= 0):
        raise ValueError(
            "vertices and padding must define a finite voxel with positive "
            "size along every axis"
        )
    n_membrane_faces = len(faces)
    if not periodic:
        voxel_vertices, voxel_faces = _aabb_to_mesh(np.zeros(3), voxel_size)
        faces = np.vstack((faces, voxel_faces + len(vertices)))
        vertices = np.vstack((vertices, voxel_vertices))
    return vertices, faces, voxel_size, n_membrane_faces, shift


def _filter_mesh_lookup(
    triangle_indices, subvoxel_indices, n_faces, excluded_faces=None
):
    """Return a CSR lookup after excluding voxel walls and selected faces."""
    keep = triangle_indices < n_faces
    if excluded_faces is not None:
        kept_positions = np.flatnonzero(keep)
        keep[kept_positions] = ~excluded_faces[triangle_indices[kept_positions]]
    kept_prefix = np.empty(triangle_indices.size + 1, dtype=np.int64)
    kept_prefix[0] = 0
    np.cumsum(keep, dtype=np.int64, out=kept_prefix[1:])
    return triangle_indices[keep], kept_prefix[subvoxel_indices]


def _classification_mesh_lookup(substrate):
    """Exclude voxel walls and lower-seam faces from +x parity rays.

    The collision lookup repeats periodically. A membrane face lying at x=0
    would otherwise be seen again at x=voxel_size, the endpoint of a parity
    ray. That periodic copy is outside the fundamental voxel; the original
    x=0 face is behind every interior query point and is not needed either.
    """
    n_faces = substrate.n_membrane_faces
    face_x = substrate.vertices[substrate.faces[:n_faces], 0]
    lower_seam_face = np.all(face_x == 0, axis=1)
    return _filter_mesh_lookup(
        substrate.triangle_indices,
        substrate.subvoxel_indices,
        n_faces,
        excluded_faces=lower_seam_face,
    )


def _pack_triangle_geometry(vertices, faces):
    """Pack triangle origins, edges, and unit normals for GPU intersection."""
    origins = vertices[faces[:, 0]]
    edge_1 = vertices[faces[:, 1]] - origins
    edge_2 = vertices[faces[:, 2]] - origins
    normals = np.cross(edge_1, edge_2)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return np.ascontiguousarray(
        np.column_stack((origins, edge_1, edge_2, normals)), dtype=np.float64
    )


@numba.jit(nopython=True)
def _cross_product(a, b):
    """Compiled function for calculating the cross product between two 1D
    arrays of length 3."""
    c = np.zeros(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


@numba.jit(nopython=True)
def _dot_product(a, b):
    """Compiled function for calculating the dot product between two 1D arrays
    of length 3."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@numba.jit(nopython=True)
def _triangle_box_overlap(triangle, box):
    """Check if a triangle overlaps with a box.

    Parameters
    ----------
    triangle : numpy.ndarray
        Array with shape (3, 3) where the first dimension corresponds to the
        points of the triangle.
    box : numpy.ndarray
        Array with shape (2, 3) so that box[0, :] and box[1, :] are the corners
        closest to and furthest from the origin.

    Returns
    -------
    bool

    Notes
    -----
    This function is based on an algorithm by Tomas Akenine-Möller presented
    in the manuscript "More Fast 3D Triangle-Box Overlap Testing".
    """

    # Move the box and triangle so that the box's centre is at the origin
    c = np.array([np.mean(box[:, i]) for i in range(3)])
    h = np.abs(box[1] - box[0]) / 2
    v = triangle - c
    e = np.eye(3)

    # Test the triangle AABB against the box
    box_aabb = np.array(
        [[np.min(v[:, i]) for i in range(3)], [np.max(v[:, i]) for i in range(3)]]
    )
    if np.all(box_aabb[0] > h) or np.all(box_aabb[1] < -h):
        return False

    # Test the plane in which the triangle is against the box
    f = np.array(
        [
            [v[1, 0] - v[0, 0], v[1, 1] - v[0, 1], v[1, 2] - v[0, 2]],
            [v[2, 0] - v[1, 0], v[2, 1] - v[1, 1], v[2, 2] - v[1, 2]],
            [v[0, 0] - v[2, 0], v[0, 1] - v[2, 1], v[0, 2] - v[2, 2]],
        ]
    )
    normal = _cross_product(f[0], f[1])
    corners = np.array(
        [
            [h[0], h[1], h[2]],
            [-h[0], -h[1], -h[2]],
            [-h[0], h[1], h[2]],
            [h[0], -h[1], -h[2]],
            [h[0], -h[1], h[2]],
            [-h[0], h[1], -h[2]],
            [h[0], h[1], -h[2]],
            [-h[0], -h[1], h[2]],
        ]
    )
    in_plane = False
    behind = np.zeros(8, dtype=numba.boolean)
    for i, point in enumerate(corners):
        dp = _dot_product(normal, v[0] - point)
        if dp == 0:
            in_plane = True
        else:
            behind[i] = _dot_product(normal, v[0] - point) > 0
    if not in_plane and (np.all(behind) or np.all(~behind)):
        return False

    # Test the triangle against the box
    p = np.zeros(3)
    for i in range(3):
        for j in range(3):
            a = _cross_product(e[i], f[j])
            r = _dot_product(h, np.abs(a))
            for k in range(3):
                p[k] = _dot_product(a, v[k])
            if np.min(p) > r or np.max(p) < -r:
                return False
    return True


@numba.jit(nopython=True)
def _interval_sv_overlap(xs, x1, x2):
    """Return the indices of subvoxels that overlap with interval [x1, x2].

    Parameters
    ----------
    xs : numpy.ndarray
        Array of subvoxel boundaries.
    x1 : float
        Start/end point of the interval.
    x2 : float
        End/start point of the interval.

    Returns
    -------
    ll : float
        Lowest subvoxel index of the overlapping subvoxels.
    ul : float
        Highest subvoxel index of the overlapping subvoxels.
    """
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    # A zero-width triangle AABB on an internal grid plane touches both closed
    # subvoxels.  Index it on both sides so traversal direction and floating
    # roundoff cannot make a real surface disappear from the lookup.
    scale = max(abs(xs[0]), abs(xs[-1]), abs(xmin), abs(xmax), abs(xs[-1] - xs[0]))
    tolerance = 8 * _FLOAT64_EPS * scale
    if abs(xmax - xmin) <= tolerance:
        for i in range(1, len(xs) - 1):
            if abs(xmin - xs[i]) <= tolerance:
                return i - 1, i + 1
    if xmin <= xs[0]:
        ll = 0
    elif xmin >= xs[-1]:
        ll = len(xs) - 1
    else:
        ll = 0
        for i, x in enumerate(xs):
            if x > xmin:
                ll = i - 1
                break
    if xmax >= xs[-1]:
        ul = len(xs) - 1
    elif xmax <= xs[0]:
        ul = 0
    else:
        ul = len(xs) - 1
        for i, x in enumerate(xs):
            if not x < xmax:
                ul = i
                break
    if ll != ul:
        return ll, ul
    else:
        if ll != len(xs) - 1:
            return ll, ul + 1
        else:
            return ll - 1, ul


@numba.jit(nopython=True)
def _triangle_aabb(triangle):
    """Calculate the axis-aligned bounding box of a triangle and return its
    closest and furthest points to the origin.

    Parameters
    ----------
    triangle : numpy.ndarray
        Array with shape (3, 3) where the first dimension corresponds to the
        points of the triangle.

    Returns
    -------
    numpy.ndarray
    """
    aabb = np.zeros((2, 3))
    for i in range(3):
        aabb[0, i] = np.min(triangle[:, i])
        aabb[1, i] = np.max(triangle[:, i])
    return aabb


@numba.jit(nopython=True)
def _box_subvoxel_overlap(box, xs, ys, zs):
    """Find the subvoxels which with a box overlaps and return the lowest and
    highest index of the subvoxels along each axis.

    Parameters
    ----------
    box : numpy.ndarray
        Array with shape (2, 3) so that box[0, :] and box[1, :] are the corners
        closest to and furthest from the origin.
    xs, ys, zs : numpy.ndarray
        Subvoxel boundaries along each axis.

    Returns
    -------
    numpy.ndarray
    """
    subvoxels = np.zeros((3, 2), dtype=np.int32)
    for i, a in enumerate([xs, ys, zs]):
        subvoxels[i] = _interval_sv_overlap(a, box[0, i], box[1, i])
    return subvoxels


@numba.jit(nopython=True)
def _mesh_triangle_subvoxel_pairs(vertices, faces, xs, ys, zs, ny, nz):
    """Collect overlapping subvoxel and triangle indices in face order."""
    subvoxel_ids = []
    triangle_ids = []
    for i in range(faces.shape[0]):
        triangle = vertices[faces[i]]
        subvoxels = _box_subvoxel_overlap(_triangle_aabb(triangle), xs, ys, zs)
        for x in range(subvoxels[0, 0], subvoxels[0, 1]):
            for y in range(subvoxels[1, 0], subvoxels[1, 1]):
                for z in range(subvoxels[2, 0], subvoxels[2, 1]):
                    box = np.array(
                        [[xs[x], ys[y], zs[z]], [xs[x + 1], ys[y + 1], zs[z + 1]]]
                    )
                    # The overlap predicate uses separating-axis comparisons.
                    # Expand each candidate box by a few coordinate-scale ulps
                    # so a triangle coplanar with a closed box face cannot be
                    # rejected by cancellation in those dot products.
                    for axis in range(3):
                        scale = max(
                            abs(box[0, axis]),
                            abs(box[1, axis]),
                            abs(box[1, axis] - box[0, axis]),
                        )
                        tolerance = 8 * _FLOAT64_EPS * scale
                        box[0, axis] -= tolerance
                        box[1, axis] += tolerance
                    if _triangle_box_overlap(triangle, box):
                        subvoxel_ids.append(x * ny * nz + y * nz + z)
                        triangle_ids.append(i)
    return subvoxel_ids, triangle_ids


def _mesh_space_subdivision(vertices, faces, voxel_size, n_sv):
    """Divide the voxel into subvoxels and return arrays for finding the
    triangles in given a subvoxel.

    Parameters
    ----------
    vertices : numpy.ndarray
        Floating-point array with shape (number of vertices, 3) containing the
        vertices of the triangular mesh.
    faces : numpy ndarray
        Integer array with shape (number of triangles, 3) containing the vertex
        indices of the points of the triangles.
    voxel_size : numpy.ndarray
        Floating-point array with shape (3,).
    n_sv : numpy.ndarray
        Integer array of size (3,) defining the number of subvoxels along each
        axis.

    Returns
    -------
    xs : numpy.ndarray
        Floating-point array with shape (n_sv[0],) containing the subvoxel
        boundaries along the x-axis.
    ys : numpy.ndarray
        Floating-point array with shape (n_sv[1],) containing the subvoxel
        boundaries along the y-axis.
    zs : numpy.ndarray
        Floating-point array with shape (n_sv[2],) containing the subvoxel
        boundaries along the z-axis.
    triangle_indices : numpy.ndarray
        One-dimensional integer array containing the triangle indices for all
        subvoxels.
    subvoxel_indices : numpy.ndarray
        Two-dimensional integer array that enables the triangles of a given
        subvoxel to be located in triangle_indices. The triangles in subvoxel i
        are the elements from subvoxel_indices[i, 0] to subvoxel_indices[i, 1].
    """

    # Define the subvoxel boundaries
    xs = np.linspace(0, voxel_size[0], n_sv[0] + 1)
    ys = np.linspace(0, voxel_size[1], n_sv[1] + 1)
    zs = np.linspace(0, voxel_size[2], n_sv[2] + 1)
    subvoxel_ids, triangle_ids = _mesh_triangle_subvoxel_pairs(
        vertices, faces, xs, ys, zs, n_sv[1], n_sv[2]
    )
    subvoxel_ids = np.asarray(subvoxel_ids, dtype=int)
    triangle_ids = np.asarray(triangle_ids, dtype=int)
    # Stable sorting retains the original face order within each subvoxel.
    triangle_indices = triangle_ids[np.argsort(subvoxel_ids, kind="stable")]
    counts = np.bincount(subvoxel_ids, minlength=np.prod(n_sv))
    ends = np.cumsum(counts)
    subvoxel_indices = np.column_stack((ends - counts, ends))
    return xs, ys, zs, triangle_indices, subvoxel_indices


@numba.njit(cache=True)
def _sampling_surface_scale(vertices, faces, n_faces, centre):
    """Return area and enclosed signed volume without allocating face arrays."""
    twice_area = 0.0
    six_times_volume = 0.0
    for face_id in range(n_faces):
        p0 = vertices[faces[face_id, 0]]
        p1 = vertices[faces[face_id, 1]]
        p2 = vertices[faces[face_id, 2]]
        ax = p1[0] - p0[0]
        ay = p1[1] - p0[1]
        az = p1[2] - p0[2]
        bx = p2[0] - p0[0]
        by = p2[1] - p0[1]
        bz = p2[2] - p0[2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        twice_area += np.sqrt(cx * cx + cy * cy + cz * cz)
        six_times_volume += (
            (p0[0] - centre[0]) * cx
            + (p0[1] - centre[1]) * cy
            + (p0[2] - centre[2]) * cz
        )
    return twice_area / 2.0, abs(six_times_volume) / 6.0


@numba.njit(cache=True)
def _mark_sampling_triangle_aabbs(vertices, faces, n_faces, shape, cell_size, mask):
    """Mark every cell intersecting a membrane triangle's conservative AABB."""
    ny, nz = shape[1], shape[2]
    for face_id in range(n_faces):
        lower = np.empty(3, dtype=np.int64)
        upper = np.empty(3, dtype=np.int64)
        for axis in range(3):
            a = vertices[faces[face_id, 0], axis]
            b = vertices[faces[face_id, 1], axis]
            c = vertices[faces[face_id, 2], axis]
            lo = min(a, b, c)
            hi = max(a, b, c)
            # Include triangles on grid planes on both sides. The tolerance
            # also protects the floor operation from roundoff in coordinates.
            tolerance = (
                16 * _FLOAT64_EPS * max(abs(lo), abs(hi), cell_size[axis] * shape[axis])
            )
            lower[axis] = max(0, int(np.floor((lo - tolerance) / cell_size[axis])))
            upper[axis] = min(
                shape[axis] - 1,
                int(np.floor((hi + tolerance) / cell_size[axis])),
            )
        for x in range(lower[0], upper[0] + 1):
            for y in range(lower[1], upper[1] + 1):
                start = x * ny * nz + y * nz
                for z in range(lower[2], upper[2] + 1):
                    mask[start + z] = 1


@numba.njit(cache=True)
def _mark_sampling_exterior(shape, mask):
    """Flood untouched cells from the six voxel faces (six connectivity)."""
    nx, ny, nz = shape
    plane = ny * nz
    queue = np.empty(mask.size, dtype=np.int32)
    tail = 0
    for cell in range(mask.size):
        x = cell // plane
        y = (cell // nz) % ny
        z = cell % nz
        if mask[cell] == 0 and (
            x == 0 or x == nx - 1 or y == 0 or y == ny - 1 or z == 0 or z == nz - 1
        ):
            mask[cell] = 2
            queue[tail] = cell
            tail += 1
    head = 0
    while head < tail:
        cell = queue[head]
        head += 1
        x = cell // plane
        y = (cell // nz) % ny
        z = cell % nz
        if x > 0 and mask[cell - plane] == 0:
            mask[cell - plane] = 2
            queue[tail] = cell - plane
            tail += 1
        if x + 1 < nx and mask[cell + plane] == 0:
            mask[cell + plane] = 2
            queue[tail] = cell + plane
            tail += 1
        if y > 0 and mask[cell - nz] == 0:
            mask[cell - nz] = 2
            queue[tail] = cell - nz
            tail += 1
        if y + 1 < ny and mask[cell + nz] == 0:
            mask[cell + nz] = 2
            queue[tail] = cell + nz
            tail += 1
        if z > 0 and mask[cell - 1] == 0:
            mask[cell - 1] = 2
            queue[tail] = cell - 1
            tail += 1
        if z + 1 < nz and mask[cell + 1] == 0:
            mask[cell + 1] = 2
            queue[tail] = cell + 1
            tail += 1


def _mesh_sampling_grid(substrate, intra=True, max_bytes=_SAMPLING_GRID_MAX_BYTES):
    """Build a conservative equal-volume mesh-compartment proposal grid.

    The grid is independent of the collision subvoxels. All membrane AABB
    cells are retained. For inside sampling, all untouched cells not reachable
    from the voxel boundary are also retained. Outside sampling uses the
    whole-voxel proposal: an untouched cell cut off from the boundary may be
    an outside cavity between nested shells or across a narrow passage. The
    GPU parity classifier always makes the final decision. ``max_bytes``
    bounds dense scratch space.
    """
    if not isinstance(max_bytes, int) or max_bytes < 32:
        raise ValueError("max_bytes must be an integer of at least 32")
    if not intra:
        return None
    max_cells = min(max_bytes // 32, np.iinfo(np.int32).max)
    voxel_size = np.asarray(substrate.voxel_size, dtype=np.float64)
    area, volume = _sampling_surface_scale(
        substrate.vertices,
        substrate.faces,
        substrate.n_membrane_faces,
        voxel_size / 2,
    )
    # 4V/A is the diameter of a long cylinder. It chooses a physical mesh
    # scale without using the separate collision-grid resolution as a knob.
    width = 4 * volume / area if area > 0 else 0.0
    if not np.isfinite(width) or width <= 0:
        width = min(voxel_size)
    # Keep the initial integer shape safely bounded even for meshes whose
    # signed volumes cancel between separate components.
    width = max(
        width,
        np.cbrt(np.prod(voxel_size) / max_cells),
        max(voxel_size) / max_cells,
    )

    def grid_shape(cell_width):
        return np.maximum(1, np.ceil(voxel_size / cell_width).astype(np.int64))

    shape = grid_shape(width)
    if np.prod(shape, dtype=np.int64) > max_cells:
        lower_width = width
        upper_width = max(voxel_size)
        for _ in range(48):
            middle = (lower_width + upper_width) / 2
            if np.prod(grid_shape(middle), dtype=np.int64) > max_cells:
                lower_width = middle
            else:
                upper_width = middle
        shape = grid_shape(upper_width)
    shape = np.ascontiguousarray(shape, dtype=np.int32)
    cell_size = np.ascontiguousarray(voxel_size / shape, dtype=np.float64)
    mask = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.uint8)
    _mark_sampling_triangle_aabbs(
        substrate.vertices,
        substrate.faces,
        substrate.n_membrane_faces,
        shape,
        cell_size,
        mask,
    )
    _mark_sampling_exterior(shape, mask)
    cell_ids = np.flatnonzero(mask != 2).astype(np.int32)
    return cell_ids, shape, cell_size


def _aabb_to_mesh(a, b):
    """Return a triangular mesh that corresponds to an axis-aligned bounding
    box defined by points a and b."""
    vertices = np.array(
        [
            [a[0], a[1], a[2]],
            [b[0], a[1], a[2]],
            [b[0], b[1], a[2]],
            [b[0], b[1], b[2]],
            [a[0], b[1], b[2]],
            [a[0], a[1], b[2]],
            [a[0], b[1], a[2]],
            [b[0], a[1], b[2]],
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
    return vertices, faces
