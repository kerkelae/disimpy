"""This module contains general useful functions."""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def vec2vec_rotmat(v, k):
    """Return a rotation matrix defining a rotation that aligns v with k.

    Parameters
    -----------
    v : numpy.ndarray
        1D array with length 3.
    k : numpy.ndarray
        1D array with length 3.

    Returns
    ---------
    R : numpy.ndarray
        3 by 3 rotation matrix.
    """
    v = v / np.linalg.norm(v)
    k = k / np.linalg.norm(k)
    axis = np.cross(v, k)
    if np.linalg.norm(axis) < np.finfo(float).eps:
        if np.linalg.norm(v - k) > np.linalg.norm(v):
            return -np.eye(3)
        else:
            return np.eye(3)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(v, k))
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = (
        np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.matmul(K, K)
    )  # Rodrigues' rotation formula
    return R


def show_traj(traj_file):
    """Plot walker trajectories saved in a trajectories file.

    Parameters
    ----------
    traj_file : str
        Path of a trajectories file where every line represents a time point
        and every line contains the positions as follows: walker_1_x walker_1_y
        walker_1_z walker_2_x walker_2_y walker_2_z...

    Returns
    -------
    None
    """
    trajectories = np.loadtxt(traj_file)
    trajectories = trajectories.reshape(
        (trajectories.shape[0], int(trajectories.shape[1] / 3), 3)
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(trajectories.shape[1]):
        ax.plot(
            trajectories[:, i, 0],
            trajectories[:, i, 1],
            trajectories[:, i, 2],
            alpha=0.5,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.ticklabel_format(style="sci", scilimits=(0, 0))
    fig.tight_layout()
    plt.show()
    return


def show_mesh(substrate):
    """Visualize a triangular mesh with random triangle colours.

    Parameters
    ----------
    substrate : disimpy.substrates._Substrate
        Substrate object containing the triangular mesh.

    Returns
    -------
    None
    """
    np.random.seed(123)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The code below often resulted in a runtime warning so ignore warnings
        # as a temporary solution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for idx in substrate.faces:
            tri = Poly3DCollection(substrate.vertices[idx], alpha=0.5)
            tri.set_facecolor(np.random.random(3))
            ax.add_collection3d(tri)
        ax.set_xlim([0, substrate.voxel_size[0]])
        ax.set_ylim([0, substrate.voxel_size[1]])
        ax.set_zlim([0, substrate.voxel_size[2]])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.ticklabel_format(style="sci", scilimits=(0, 0))
        fig.tight_layout()
        plt.show()
    return
