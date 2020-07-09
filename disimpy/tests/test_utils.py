"""This module contains unit tests of the utils module."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from .. import utils


def test_vec2vec_rotmat():
    a = np.array([1, 0, 0])
    for b in np.array([[0, 0, 1], [-1, 0, 0], [1, 0, 0]]):
        R = utils.vec2vec_rotmat(a, b)
        npt.assert_array_almost_equal(np.dot(R, a), b)


def test_show_traj():
    traj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_traj.txt')
    utils.show_traj(traj_file, show=False)
    plt.close('all')
    return
