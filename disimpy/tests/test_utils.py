"""This module contains unit tests of the utils module."""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.testing as npt

from .. import utils


def test_vec2vec_rotmat():
    a = np.array([1, 0, 0])
    for b in np.array([[0, 0, 1], [-1, 0, 0], [1, 0, 0]]):
        R = utils.vec2vec_rotmat(a, b)
        assert_array_almost_equal(np.dot(R, a), b)


def test_show_traj():
    traj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'example_traj.txt')
    fig, ax = utils.show_traj(traj_file, title=None, show=False)
    npt.assert_equal(isinstance(fig, matplotlib.figure.Figure), True)
    plt.close(fig)
    return
