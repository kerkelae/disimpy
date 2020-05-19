"""This module contains general utility functions."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vec2vec_rotmat(v, k):
    """Return rotation matrix defining a rotation that aligns v with k.
    
    Parameters
    -----------
    v : array_like
        3D vector.
    k : array_like
        3D vector.

    Returns
    ---------
    R : ndarray
        3 by 3 rotation matrix.
    """
    axis = np.cross(v, k).astype(np.float)
    axis /= np.linalg.norm(axis)
    angle = np.arcsin(np.linalg.norm(np.cross(v, k)) /
                      (np.linalg.norm(k) * np.linalg.norm(v)))
    if np.dot(v, k) < 0:
        angle = np.pi - angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + \
        np.sin(angle) * K + \
        (1 - np.cos(angle)) * np.matmul(K, K)  # Rodrigues' rotation formula
    return R


def show_traj(traj_file, title=None, show=True):
    """Visualize walker trajectories in a trajectories file.

    Parameters
    ----------
    traj_file : str
        Path to trajectories file that contains walker trajectories. Every line
        represents a time point. Every line contains the positions as follows:
        walker_1_x walker_1_y walker_1_z walker_2_x walker_2_y walker_2_z...
    title : str, optional.
        Title of the figure.
    show : bool
        Boolean switch defining whether to render figure or not.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes._subplots.Axes3DSubplot
    """
    trajectories = np.loadtxt(traj_file)
    trajectories = trajectories.reshape((trajectories.shape[0],
                                         int(trajectories.shape[1] / 3),
                                         3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(trajectories.shape[1]):
        ax.plot(trajectories[:, i, 0],
                trajectories[:, i, 1],
                trajectories[:, i, 2],
                alpha=.5)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-np.max(np.abs(trajectories)), np.max(np.abs(trajectories))])
    ax.set_ylim([-np.max(np.abs(trajectories)), np.max(np.abs(trajectories))])
    ax.set_zlim([-np.max(np.abs(trajectories)), np.max(np.abs(trajectories))])
    ax.ticklabel_format(style='sci', scilimits=(0, 0))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
