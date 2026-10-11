"""This module contains general useful functions."""

import numpy as np


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
    v = np.asarray(v, dtype=float)
    k = np.asarray(k, dtype=float)
    if v.shape != (3,) or k.shape != (3,):
        raise ValueError("v and k must be three-dimensional vectors")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(k)):
        raise ValueError("v and k must be finite, nonzero vectors")
    v_scale = np.max(np.abs(v))
    k_scale = np.max(np.abs(k))
    if v_scale == 0 or k_scale == 0:
        raise ValueError("v and k must be finite, nonzero vectors")
    # Scale before taking the norm so very large and subnormal finite vectors
    # can be normalized without overflow or underflow.
    v = v / v_scale
    k = k / k_scale
    v_norm = np.linalg.norm(v)
    k_norm = np.linalg.norm(k)
    v = v / v_norm
    k = k / k_norm
    axis = np.cross(v, k)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(v, k), -1.0, 1.0))
    if axis_norm <= 8 * np.finfo(float).eps:
        if dot < 0:
            # A reflection (-I) maps v to -v but is not a rotation. Rotate
            # around any deterministic axis perpendicular to v instead.
            basis = np.zeros(3)
            basis[np.argmin(np.abs(v))] = 1
            axis = np.cross(v, basis)
            axis /= np.linalg.norm(axis)
            return 2 * np.outer(axis, axis) - np.eye(3)
        else:
            return np.eye(3)
    axis /= axis_norm
    angle = np.arccos(dot)
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
    import matplotlib.pyplot as plt

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


def show_mesh(substrate, seed=123):
    """Visualize a triangular mesh with random triangle colours.

    Parameters
    ----------
    substrate : disimpy.substrates._Substrate
        Substrate object containing the triangular mesh.
    seed : int, optional
        Seed for pseudorandom number generation.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    np.random.seed(seed)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for idx in substrate.faces:
        tri = Poly3DCollection([substrate.vertices[idx]], alpha=0.5)
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
