"""This module contains tests of the utils module."""

import os
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from .. import utils


def test_vec2vec_rotmat():
    np.random.seed(123)
    for _ in range(100):
        a = np.random.random(3) - .5
        a_norm = np.linalg.norm(a)
        b = np.random.random(3) - .5
        b_norm = np.linalg.norm(b)
        R = utils.vec2vec_rotmat(a, b)
        npt.assert_array_almost_equal(np.linalg.norm(a), a_norm)
        npt.assert_array_almost_equal(np.linalg.norm(b), b_norm)
        a = np.matmul(R, a)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        npt.assert_array_almost_equal(a, b)
    return