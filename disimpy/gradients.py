"""This module contains code for creating and manipulating gradient arrays.

Gradient arrays are numpy.ndarray instances of shape (number of measurements,
number of time points, 3). Gradient array elements are floats representing the
gradient magnitude in SI units (T/m).
"""

import numpy as np
import scipy.integrate

from .simulations import GAMMA


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
        interp_g[m, :, 0] = np.interp(np.arange(n_t), np.linspace(
            0, n_t - 1, gradient.shape[1]), gradient[m, :, 0])
        interp_g[m, :, 1] = np.interp(np.arange(n_t), np.linspace(
            0, n_t - 1, gradient.shape[1]), gradient[m, :, 1])
        interp_g[m, :, 2] = np.interp(np.arange(n_t), np.linspace(
            0, n_t - 1, gradient.shape[1]), gradient[m, :, 2])
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
    q = GAMMA * scipy.integrate.cumtrapz(gradient, axis=1, dx=dt, initial=0)
    return q


def calc_b(gradient, dt):
    """Calculate b-value(s) of the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.

    Returns
    -------
    q : numpy.ndarray
        b-value(s).
    """
    q = calc_q(gradient, dt)
    b = np.trapz(np.linalg.norm(q, axis=2)**2, axis=1, dx=dt)
    return b


def set_b(gradient, dt, b):
    """Scale the gradient array magnitude to correspond to given b-value(s).

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
    g : numpy.ndarray
        Scaled gradient array with magnitude equal to given b-value(s).
    """
    if np.any(b == 0):
        raise ValueError('b can not be equal to 0. If b = 0, the simulated'
                         + 'signal is simply equal to the number of random'
                         + ' walkers.')
    ratio = b / calc_b(gradient, dt)
    gradient *= np.sqrt(ratio)[:, np.newaxis, np.newaxis]
    return gradient


def rotate_gradient(gradient, Rs):
    """Rotate the gradient array with rotation matrix array. The gradient array
    of each measurement is rotated by the corresponding rotation matrix.

    Parameters
    ----------
    gradient : ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    Rs : ndarray
        Array of rotation matrices of shape (n of measurements, 3, 3).

    Returns
    -------
    g : ndarray
        Rotated gradient array.
    """
    g = np.zeros(gradient.shape)
    for m, R in enumerate(Rs):
        g[m, :, :] = np.matmul(R, gradient[m, :, :].T).T
    return g
