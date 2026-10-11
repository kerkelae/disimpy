"""This module contains tests of the utils module."""

import numpy as np
import numpy.testing as npt

from .. import utils


SEED = 123


def test_vec2vec_rotmat():
    np.random.seed(SEED)
    for _ in range(1000):
        a = np.random.random(3) - 0.5
        a_norm = np.linalg.norm(a)
        b = np.random.random(3) - 0.5
        b_norm = np.linalg.norm(b)
        R = utils.vec2vec_rotmat(a, b)
        npt.assert_array_almost_equal(np.linalg.norm(a), a_norm)
        npt.assert_array_almost_equal(np.linalg.norm(b), b_norm)
        a = np.matmul(R, a)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        npt.assert_array_almost_equal(a, b)
    return


def test_vec2vec_rotmat_antiparallel_is_proper_rotation():
    for v in (np.array([1.0, 0.0, 0.0]), np.array([1.0, 2.0, -3.0])):
        R = utils.vec2vec_rotmat(v, -v)
        npt.assert_allclose(R @ v, -v, atol=1e-14)
        npt.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1, atol=1e-14)


def test_vec2vec_rotmat_normalizes_extreme_finite_vectors_stably():
    for scale in (np.finfo(float).max / 4, np.nextafter(0.0, 1.0)):
        v = np.array([scale, 0.0, 0.0])
        k = np.array([0.0, -scale, 0.0])
        R = utils.vec2vec_rotmat(v, k)
        npt.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-14)
        npt.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)
        npt.assert_allclose(np.linalg.det(R), 1, atol=1e-14)
