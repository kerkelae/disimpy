"""Fast checks for CPU walker initialization."""

import numpy as np
import pytest

from .. import simulations


@pytest.mark.parametrize(
    ("sample", "args", "semiaxes"),
    [
        (simulations._fill_circle, (8, 5e-6), np.full(2, 5e-6)),
        (simulations._fill_sphere, (8, 5e-6), np.full(3, 5e-6)),
        (
            simulations._fill_ellipsoid,
            (8, np.array([4e-6, 5e-6, 6e-6])),
            np.array([4e-6, 5e-6, 6e-6]),
        ),
    ],
)
def test_seeded_sampling(sample, args, semiaxes):
    simulations._set_seed(123)
    points = sample(*args)
    simulations._set_seed(123)
    np.testing.assert_array_equal(points, sample(*args))
    assert points.shape == (8, len(semiaxes))
    assert np.all(np.linalg.norm(points / semiaxes, axis=1) < 1)


def test_rotated_initial_positions():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    semiaxes = np.array([4e-6, 5e-6, 6e-6])

    simulations._set_seed(123)
    cylinder = simulations._initial_positions_cylinder(8, 5e-6, rotation)
    local_cylinder = cylinder @ rotation
    np.testing.assert_array_equal(local_cylinder[:, 0], np.zeros(8))
    assert np.all(np.linalg.norm(local_cylinder[:, 1:3], axis=1) < 5e-6)

    simulations._set_seed(123)
    ellipsoid = simulations._initial_positions_ellipsoid(8, semiaxes, rotation)
    local_ellipsoid = ellipsoid @ rotation
    assert np.all(np.linalg.norm(local_ellipsoid / semiaxes, axis=1) < 1)
