import numpy as np
import scipy.integrate

"""This is the module for manipulating gradient arrays.

Gradient arrays are numpy.ndarray instances of shape (number of measurements,
number of time points, 3). Array elements are floats representing the gradient
magnitude in SI units (T/m).
"""


def interpolate_gradient(gradient, dt, n_t):
    """Interpolate gradient array to have n_t time points.
    
    Parameters
    ----------
    gradient : ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.
    n_t : int
        The number of time points in gradient array after interpolation.
    
    Returns
    -------
    interp_g : ndarray
        Interpolated gradient array.
    dt : float
        Duration of time step in interp_g.
    """
    dt *= gradient.shape[1]/n_t
    interp_g = np.zeros((gradient.shape[0], int(n_t), gradient.shape[2]))
    for m in range(gradient.shape[0]):
        interp_g[m, :, 0] = np.interp(np.arange(n_t),
                                      np.linspace(0, n_t-1, gradient.shape[1]),
                                      gradient[m, :, 0])
        interp_g[m, :, 1] = np.interp(np.arange(n_t),
                                      np.linspace(0, n_t-1, gradient.shape[1]),
                                      gradient[m, :, 1])
        interp_g[m, :, 2] = np.interp(np.arange(n_t),
                                      np.linspace(0, n_t-1, gradient.shape[1]),
                                      gradient[m, :, 2])
    return interp_g, dt


def calc_q(gradient, dt):
    """Calculate q-vector(s) corresponding to gradient array.
    
    Parameters
    ----------
    gradient : ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.
    
    Returns
    -------
    q : ndarray
        q-vector(s).  
    """
    GAMMA = 267.513e6
    q = GAMMA*scipy.integrate.cumtrapz(gradient, axis=1, dx=dt, initial=0)
    return q


def calc_b(gradient, dt):
    """Calculate b-value(s) corresponding to gradient array.
    
    Parameters
    ----------
    gradient : ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.
    
    Returns
    -------
    q : array_like
        b-values.
    """
    q = calc_q(gradient, dt)
    b = np.trapz(np.linalg.norm(q, axis=2)**2, axis=1, dx=dt)
    return b
        
def set_b(gradient, dt, b):
    """Calculate gradient array of given b-value(s)
    
    Parameters
    ----------
    gradient : ndarray
        Gradient array of shape (n of measurements, n of time points, 3).
    dt : float
        Duration of time step in gradient array.
    b : array_like
        b-value or a list of b-values.
    
    Returns
    -------
    g : ndarray
        Gradient array with chosen b-value(s).
    """
    ratio = b / calc_b(gradient, dt)
    max_G = np.max(np.linalg.norm(gradient, axis=2), axis=1)
    g = gradient / max_G[:, np.newaxis, np.newaxis]
    #g[np.isnan(g)] = 0
    g *= (np.sqrt(ratio)*max_G)[:,np.newaxis,np.newaxis]
    return g

def rotate_gradient(gradient, Rs):
    """Rotate gradients with rotation matrices

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
        g[m, : , :] = np.matmul(R, gradient[m, :, :].T).T
    return g
