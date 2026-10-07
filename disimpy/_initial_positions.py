"""CPU sampling of initial walker positions."""

import numba
import numpy as np


@numba.jit(nopython=True)
def _set_seed(seed):
    """Set the pseudorandom number generator seed for compiled functions."""
    np.random.seed(seed)
    return


@numba.jit(nopython=True)
def _fill_circle(n, radius):
    """Sample n random points from a uniform distribution inside a circle."""
    i = 0
    filled = False
    points = np.zeros((n, 2))
    while not filled:
        p = (np.random.random(2) - 0.5) * 2 * radius
        if np.linalg.norm(p) < radius:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


@numba.jit(nopython=True)
def _fill_sphere(n, radius):
    """Sample n random points from a uniform distribution inside a sphere."""
    i = 0
    filled = False
    points = np.zeros((n, 3))
    while not filled:
        p = (np.random.random(3) - 0.5) * 2 * radius
        if np.linalg.norm(p) < radius:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


@numba.jit(nopython=True)
def _fill_ellipsoid(n, semiaxes):
    """Sample n random points from a uniform distribution inside an axis
    aligned ellipsoid with semi-axes a, b, and c."""
    filled = False
    points = np.zeros((n, 3))
    i = 0
    while not filled:
        p = (np.random.random(3) - 0.5) * 2 * semiaxes
        if np.sum((p / semiaxes) ** 2) < 1:
            points[i] = p
            i += 1
            if i == n:
                filled = True
    return points


def _initial_positions_cylinder(n_walkers, radius, R):
    """Calculate initial positions for spins in a cylinder whose orientation is
    defined by R which defines the rotation from cylinder frame to lab
    frame."""
    positions = np.zeros((n_walkers, 3))
    positions[:, 1:3] = _fill_circle(n_walkers, radius)
    positions = np.matmul(R, positions.T).T
    return positions


def _initial_positions_ellipsoid(n_walkers, semiaxes, R):
    """Calculate initial positions for spins in an ellipsoid with semi-axes a,
    b, c whos whose orientation is defined by R which defines the rotation from
    ellipsoid frame to lab frame."""
    positions = _fill_ellipsoid(n_walkers, semiaxes)
    positions = np.matmul(R, positions.T).T
    return positions
