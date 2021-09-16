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


def test_show_traj():
    traj_file = os.path.join(
        os.path.dirname(utils.__file__), 'tests', 'test_traj.txt')
    utils.show_traj(traj_file, show=False)
    plt.close('all')
    return


def test_show_mesh():
    mesh_file = os.path.join(
        os.path.dirname(utils.__file__), 'tests', 'cyl_mesh_r5um_l25um.npy')
    mesh = np.load(mesh_file)
    utils.show_mesh(mesh, show=False)
    plt.close('all')
    return
