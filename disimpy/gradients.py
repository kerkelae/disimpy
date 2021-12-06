"""This module contains code for creating and manipulating gradient arrays.

Gradient arrays are numpy.ndarray instances with shape (number of measurements,
number of time points, 3). Gradient array elements are floats representing the
gradient magnitude in SI units (T/m).
"""

import numpy as np


GAMMA = 267.513e6  # Gyromagnetic ratio of the simulated spins


def _cumtrapz(y, dx, axis, initial):
    """Cumulatively integrate y(x) using the composite trapezoidal rule.
    
    Parameters
    ----------
    y : numpy.ndarray
        Values to integrate.
    dx : float
        Spacing between elements of `y`.
    axis : int
        Specifies the axis to cumulate. 
    initial : scalar
        Insert this value at the beginning of the returned result.

    Returns
    -------
    res : numpy.ndarray
        The result of cumulative integration of `y` along `axis`.
    """

    # This code is directly copied from Scipy so that it can be removed from
    # dependencies

    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    y = np.asarray(y)
    d = dx
    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
    shape = list(res.shape)
    shape[axis] = 1
    res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res], axis=axis)
    return res


def interpolate_gradient(gradient, dt, n_t):
    """Interpolate the gradient array to have n_t time points.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.
    n_t : int
        The number of time points in gradient array after interpolation.

    Returns
    -------
    interp_g : numpy.ndarray
        Interpolated gradient array.
    dt : float
        Duration of time step in the interpolated gradient array.
    """
    dt *= gradient.shape[1] / n_t
    interp_g = np.zeros((gradient.shape[0], int(n_t), gradient.shape[2]))
    for m in range(gradient.shape[0]):
        interp_g[m, :, 0] = np.interp(
            np.arange(n_t),
            np.linspace(0, n_t - 1, gradient.shape[1]),
            gradient[m, :, 0],
        )
        interp_g[m, :, 1] = np.interp(
            np.arange(n_t),
            np.linspace(0, n_t - 1, gradient.shape[1]),
            gradient[m, :, 1],
        )
        interp_g[m, :, 2] = np.interp(
            np.arange(n_t),
            np.linspace(0, n_t - 1, gradient.shape[1]),
            gradient[m, :, 2],
        )
    return interp_g, dt


def calc_q(gradient, dt):
    """Calculate the q-vector array corresponding to the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.

    Returns
    -------
    q : numpy.ndarray
        q-vector array.
    """
    q = GAMMA * _cumtrapz(gradient, axis=1, dx=dt, initial=0)
    return q


def calc_b(gradient, dt):
    """Calculate b-values of the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.

    Returns
    -------
    b : numpy.ndarray
        b-values.
    """
    q = calc_q(gradient, dt)
    b = np.trapz(np.linalg.norm(q, axis=2) ** 2, axis=1, dx=dt)
    return b


def set_b(gradient, dt, b):
    """Scale the gradient array magnitude to correspond to given b-values.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.
    b : float or numpy.ndarray
        b-value or an array of b-values with length equal to n of measurements.

    Returns
    -------
    scaled_g : numpy.ndarray
        Scaled gradient array.
    """
    if np.any(b == 0):
        raise ValueError(
            "b can not be equal to 0. If b = 0, the simulated signal is simply"
            + " equal to the number of random walkers."
        )
    ratio = b / calc_b(gradient, dt)
    scaled_g = gradient * np.sqrt(ratio)[:, np.newaxis, np.newaxis]
    return scaled_g


def rotate_gradient(gradient, Rs):
    """Rotate the gradient array of each measurement according to the
    corresponding rotation matrix.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    Rs : numpy.ndarray
        Rotation matrix array of shape (n of measurements, 3, 3).

    Returns
    -------
    g : numpy.ndarray
        Rotated gradient array.
    """
    g = np.zeros(gradient.shape)
    for i, R in enumerate(Rs):
        if not np.isclose(np.linalg.det(R), 1) or not np.all(
            np.isclose(R.T, np.linalg.inv(R))
        ):
            raise ValueError("Rs[%s] (%s) is not a valid rotation matrix" % (i, R))
        g[i, :, :] = np.matmul(R, gradient[i, :, :].T).T
    return g
