import numpy as np
import numpy.testing as npt


def _non_overlapping_circles(C1, r1, C2, r2):
    """Check if two circles do not overlap.

    Parameters
    ----------
    C1 : tuple
        Center of circle C1(x1, y1).
    r1 : float
        Radius of the circle C1.
    C2 : tuple
        Center of circle C2(x2, y2).
    r2 : float
        Radius of the circle C2.

    Returns
    -------
    boolean
    """
    x1, y1 = C1
    x2, y2 = C2
    if np.sqrt((y2 - y1) ** 2 + (x1 - x2) ** 2) <= (r1 + r2):
        return False
    return True


def test__non_overlapping_circles():
    npt.assert_equal(_non_overlapping_circles((1, 2), 1, (4, 3), 1), True)
    return


def _mirrored_circles(C, r, voxel_size):
    """Create mirrored versions of a circle in the eigth surrounding voxels.
    This will add a periodic boundary condition to the central voxel.

    Parameters
    ----------
    C : tuple
        Center of circle C(x, y).
    r : float
        Radius of the circle.
    voxel_size : float
        Size of the voxel.

    Returns
    -------
    mirrors : numpy.ndarray
        Array with center coordinates for the mirrors and their radii.
    """
    x, y = C
    mirrors = np.array(
        [
            [x, y, r],
            [x - voxel_size, y, r],
            [x + voxel_size, y, r],
            [x, y - voxel_size, r],
            [x, y + voxel_size, r],
            [x - voxel_size, y - voxel_size, r],
            [x + voxel_size, y - voxel_size, r],
            [x - voxel_size, y + voxel_size, r],
            [x + voxel_size, y + voxel_size, r],
        ]
    )
    return mirrors


def test__mirrored_circles():
    mirrors = _mirrored_circles((1, 1), 1, 3)
    mirrors_test = np.array(
        [
            [1, 1, 1],
            [-2, 1, 1],
            [4, 1, 1],
            [1, -2, 1],
            [1, 4, 1],
            [-2, -2, 1],
            [4, -2, 1],
            [-2, 4, 1],
            [4, 4, 1],
        ]
    )
    npt.assert_equal(mirrors, mirrors_test)
    return


def _sampling_circles(
    n_objects,
    distribution,
):
    """Sample circles from a distribution and sort them in ascending order.

    Parameters
    ----------
    n_objects : int
        Number of circles to be sampled.
    distribution: string


    Returns
    -------
    sampled_radii : numpy.ndarray
        Array with the sampled radii.
    """
    return


def test__sampling_circles():
    return


def _place_circles(voxel_size, max_iterations=1e2):
    """Pack non-overlapping circles in a voxel respecting the periodic boundary condition.

    Parameters
    ----------
    voxel_size : float
        Size of the voxel.
    max_iterations : float
        Maximum number of iterations to place a circle in the voxel. Default is equal to 1e2.

    Returns
    -------
    placed_circles : numpy.ndarray
        Array with the center coordinates and radii of the cirlces placed in the voxel.
    placed_mirrors : numpy.ndarray
        Array with the center coordinates and radii of the cirlces placed in the voxel and their mirrored versions
        in the surrounding voxels.
    """
    return


def test__placing_circles():
    return


def _sampling_circles(n_objects, shape, scale, voxel_size, max_iterations=1e2):
    """Sample circles from a gamma distribution and packed them in a voxel with periodic boundaries.

    Parameters
    ----------
    n_objects : int
        Number of circles to be sampled.
    shape : float
        The shape of the gamma distribution.
    scale : float
        The scale of the gamma distribution.
    voxel_size : float
        Size of the voxel.
    max_iterations : float
        Maximum number of iterations to place a circle in the voxel. Default is equal to 1e2.

    Returns
    -------
    placed_circles : numpy.ndarray
        Array with the center coordinates and radii of the cirlces placed in the voxel.
    placed_mirrors : numpy.ndarray
        Array with the center coordinates and radii of the cirlces placed in the voxel and their mirrored versions
        in the surrounding voxels.
    """
    sampled_radii = np.random.gamma(shape, scale, n_objects)
    sampled_radii = np.sort(sampled_radii)[::-1]
    placed_circles = np.zeros((n_objects, 3))
    placed_mirrors = np.zeros((n_objects * 9, 3))
    filled_positions = 1
    for r in sampled_radii:
        placed = False
        i = 0
        while not placed and i < max_iterations:
            i += 1
            x, y = np.random.random(2) * voxel_size
            mirrors = _mirrored_circles((x, y), r, voxel_size)
            if np.all(placed_circles == 0):
                placed_circles[0] = x, y, r
                for mirror, k in zip(mirrors, range(len(mirrors))):
                    placed_mirrors[k] = mirror
                placed = True
            else:
                intersects = False
                for mirror_cand in mirrors:
                    m_x, m_y, r = mirror_cand
                    end = filled_positions * 9
                    for mirror_stored in placed_mirrors[:end]:
                        ms_x, ms_y, ms_r = mirror_stored
                        if not _non_overlapping_circles(
                            (m_x, m_y), r, (ms_x, ms_y), ms_r
                        ):
                            intersects = True
                            break
            if not intersects:
                placed_circles[filled_positions] = x, y, r
                interval = filled_positions * 9
                for mirror, k in zip(mirrors, range(interval, len(mirrors) + interval)):
                    placed_mirrors[k] = mirror
                placed = True
                filled_positions += 1
    return placed_circles, placed_mirrors


def _cylinder_mesh(r, C, n_faces, h):
    """Generate a triangular mesh in the shape of a cylinder.

    Parameters
    ----------
    r: float
        Radius of the base.
    C: tuple
        Center of base C(x, y).
    n_faces: int
        Number of faces for the mesh.
    h: int
        Hight of the cylinder.

    Returns
    -------
    vertices: numpy.ndarray
        Array of vertices for the mesh.
    faces: numpy.ndarray
        Array of faces for the mesh.
    """
    vertices = []
    faces = []
    thetas = np.linspace(0, 2 * np.pi, n_faces + 1)
    for theta in thetas:
        v_base = [r * np.sin(theta) + C[0], r * np.cos(theta) + C[1], 0]
        v_h = [r * np.sin(theta) + C[0], r * np.cos(theta) + C[1], h]
        vertices.append(v_base)
        vertices.append(v_h)
    indexes = range(len(vertices) - 2)
    for i in indexes:
        t = [i, i + 1, i + 2]
        faces.append(t)
    connection_points = [
        [len(vertices) - 1, len(vertices) - 2, 1],
        [len(vertices) - 2, 0, 1],
    ]
    for p in connection_points:
        faces.append(p)
    return np.array(vertices), np.array(faces)


def test__cylinder_mesh():
    r = 1e-6
    C = (0, 0)
    n_faces = 100
    h = 10e-6
    vertices, faces = _cylinder_mesh(r, C, n_faces, h)
    npt.assert_equal(vertices.shape, ((n_faces * 2) + 2, 3))
    npt.assert_equal(faces.shape, ((n_faces * 2) + 2, 3))
    npt.assert_equal(vertices.shape, faces.shape)
    npt.assert_equal(np.all(vertices[::2] == 0), True)
    npt.assert_equal(np.all(vertices[1::2] == h), True)
    return


def packed_cylinders(n_objects, voxel_size, shape, scale, n_faces, h):
    """Create a voxel of packed cylinders generated from triangular meshes, with gamma distributed radii.
    The voxel should have periodic boundaries.

    Parameters
    ----------
    n_objects: int
        Number of cylinders to be sampled.
    voxel_size: float
        Size of the voxel.
    shape: float
        The shape of the gamma distribution.
    scale: float
        The scale of the gamma distribution.
    n_faces: int
        Number of faces for the mesh.
    h: int
        Hight of the cylinders.

    Returns
    -------
    vertices: numpy.ndarray
        Array of vertices for the meshes.
    faces: numpy.ndarray
        Array of faces for the meshes.
    """
    circles = _sampling_circles(n_objects, shape, scale, voxel_size)[0]
    faces = []
    vertices = []
    for base in circles:
        x, y, r = base[0], base[1], base[2]
        v, f = _cylinder_mesh(r, (x, y), n_faces, h)
        faces.append(f)
        vertices.append(v)
    return np.asarray(vertices), np.asarray(faces)


def test_packed_cylinders():
    n_objects = int(5e2)
    voxel_size = 1e-4
    shape = 3
    scale = 1e-6
    n_faces = 50
    h = 10e-6
    vertices, faces = packed_cylinders(n_objects, voxel_size, shape, scale, n_faces, h)
    npt.assert_equal(vertices.shape, (n_objects, (n_faces * 2) + 2, 3))
    npt.assert_equal(faces.shape, (n_objects, (n_faces * 2) + 2, 3))
    npt.assert_equal(vertices.shape, faces.shape)
    npt.assert_equal(np.all(vertices[::2] == 0), True)
    npt.assert_equal(np.all(vertices[1::2] == h), True)
    return
