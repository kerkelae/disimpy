"""This module contains code for creating and manipulating gradient arrays.

Gradient arrays are numpy.ndarray instances with shape (number of measurements,
number of time points, 3). Gradient array elements are floats representing the
gradient magnitude in SI units (T/m). With ``N`` samples spaced by ``dt``, the
waveform duration is ``(N - 1) * dt`` and the gradient is treated as linear
between samples.
"""

import numpy as np

from . import utils

GAMMA = 267.513e6  # Gyromagnetic ratio of the simulated spins


def _validate_gradient(gradient, dt=None):
    """Validate the shared gradient-array representation."""
    if (
        not isinstance(gradient, np.ndarray)
        or gradient.ndim != 3
        or gradient.shape[0] < 1
        or gradient.shape[1] < 2
        or gradient.shape[2] != 3
        or not np.issubdtype(gradient.dtype, np.floating)
        or not np.all(np.isfinite(gradient))
    ):
        raise ValueError(
            "gradient must be a finite floating-point array with shape "
            "(measurements, at least two time points, 3)"
        )
    if dt is not None and (
        not isinstance(dt, (float, np.floating)) or not np.isfinite(dt) or dt <= 0
    ):
        raise ValueError("dt must be a finite positive float")


def _bvalue_array(bvals, n_measurements=None):
    """Return a finite, nonnegative b-value for each measurement."""
    bvals = np.asarray(bvals, dtype=float)
    if bvals.ndim == 0:
        bvals = np.full(1 if n_measurements is None else n_measurements, bvals)
    elif bvals.ndim != 1 or bvals.size == 0:
        raise ValueError(
            "b-values must be a scalar or a nonempty one-dimensional array"
        )
    if n_measurements is not None and bvals.size != n_measurements:
        raise ValueError("there must be one b-value per measurement")
    if not np.all(np.isfinite(bvals)) or np.any(bvals < 0):
        raise ValueError("b-values must be finite and nonnegative")
    return bvals


def interpolate_gradient(gradient, dt, n_t):
    """Interpolate the gradient array to have `n_t` time points.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in the gradient array.
    n_t : int
        The number of time points in the gradient array after interpolation.

    Returns
    -------
    interp_g : numpy.ndarray
        Interpolated gradient array.
    dt : float
        Duration of a time step in the interpolated gradient array.
    """
    _validate_gradient(gradient, dt)
    if not isinstance(n_t, (int, np.integer)) or n_t < 2:
        raise ValueError("n_t must be an integer with at least two time points")
    T = dt * (gradient.shape[1] - 1)
    dt = T / (n_t - 1)
    interp_g = np.zeros((gradient.shape[0], n_t, 3))
    source_times = np.linspace(0, T, gradient.shape[1])
    target_times = np.linspace(0, T, n_t)
    for i in range(gradient.shape[0]):
        for j in range(3):
            interp_g[i, :, j] = np.interp(
                target_times,
                source_times,
                gradient[i, :, j],
            )
    return interp_g, dt


def calc_q(gradient, dt):
    """Calculate the q-vector array corresponding to the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Time between adjacent gradient samples.

    Returns
    -------
    q : numpy.ndarray
        q-vector array.
    """
    _validate_gradient(gradient, dt)
    q = GAMMA * np.concatenate(
        (
            np.zeros((gradient.shape[0], 1, 3)),
            np.cumsum(dt * (gradient[:, 1::, :] + gradient[:, 0:-1, :]) / 2, axis=1),
        ),
        axis=1,
    )
    return q


def calc_b(gradient, dt):
    """Calculate b-values of the gradient array.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in gradient array.

    Returns
    -------
    b : numpy.ndarray
        b-values.
    """
    q = calc_q(gradient, dt)
    q_terminal = q[:, -1, :]
    # On each interval G is linear, so q is quadratic. Integrate
    # |q(T)-q(t)|^2 analytically rather than applying another quadrature rule.
    a = q_terminal[:, np.newaxis, :] - q[:, :-1, :]
    b = -GAMMA * dt * gradient[:, :-1, :]
    c = -0.5 * GAMMA * dt * np.diff(gradient, axis=1)
    integral = (
        np.sum(a * a, axis=2)
        + np.sum(a * b, axis=2)
        + (2 * np.sum(a * c, axis=2) + np.sum(b * b, axis=2)) / 3
        + np.sum(b * c, axis=2) / 2
        + np.sum(c * c, axis=2) / 5
    )
    return dt * np.sum(integral, axis=1)


def set_b(gradient, dt, b):
    """Scale the gradient array magnitude to correspond to given b-values.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    dt : float
        Duration of a time step in gradient array.
    b : float or numpy.ndarray
        b-value or an array of b-values with length equal to n of measurements.

    Returns
    -------
    scaled_g : numpy.ndarray
        Scaled gradient array.
    """
    b = _bvalue_array(b, gradient.shape[0])
    original_b = calc_b(gradient, dt)
    if not np.all(np.isfinite(original_b)):
        raise ValueError("gradient must have finite b-values")
    if np.any((original_b == 0) & (b > 0)):
        raise ValueError("cannot assign a positive b-value to a zero waveform")
    ratio = np.zeros_like(b)
    nonzero = original_b > 0
    ratio[nonzero] = b[nonzero] / original_b[nonzero]
    if not np.all(np.isfinite(ratio)):
        raise ValueError("requested b-value scaling is not finite")
    scaled_g = gradient * np.sqrt(ratio)[:, np.newaxis, np.newaxis]
    return scaled_g


def rotate_gradient(gradient, Rs):
    """Rotate the gradient array of each measurement according to the
    corresponding rotation matrix.

    Parameters
    ----------
    gradient : numpy.ndarray
        Gradient array with shape (n of measurements, n of time points, 3).
    Rs : numpy.ndarray
        Rotation matrix array with shape (n of measurements, 3, 3).

    Returns
    -------
    g : numpy.ndarray
        Rotated gradient array.
    """
    _validate_gradient(gradient)
    if (
        not isinstance(Rs, np.ndarray)
        or Rs.shape != (gradient.shape[0], 3, 3)
        or not np.issubdtype(Rs.dtype, np.floating)
        or not np.all(np.isfinite(Rs))
    ):
        raise ValueError("Rs must have shape (number of measurements, 3, 3)")
    g = np.zeros(gradient.shape)
    for i, R in enumerate(Rs):
        if not np.isclose(np.linalg.det(R), 1) or not np.allclose(R.T @ R, np.eye(3)):
            raise ValueError(f"Rs[{i}] ({R}) is not a valid rotation matrix")
        g[i, :, :] = np.matmul(R, gradient[i, :, :].T).T
    return g


def pgse(delta, DELTA, n_t, bvals, bvecs):
    """Generate a pulsed gradient spin echo gradient array.

    Parameters
    ----------
    delta: float
        Diffusion encoding time.
    DELTA : float
        Diffusion time.
    n_t : int
        The number of time points in the generated gradient array.
    bvals : float or numpy.ndarray
        b-value or an array of b-values.
    bvecs : numpy.ndarray
        b-vector or array of b-vectors. A zero vector is allowed only for b=0.

    Returns
    -------
    gradient : numpy.ndarray
        Gradient array.
    dt : float
        Duration of a time step in the gradient array.

    Raises
    ------
    ValueError
        If the parameters are invalid or either pulse has fewer than two
        nonzero samples on the requested time grid.
    """
    bvals = _bvalue_array(bvals)
    bvecs = np.asarray(bvecs, dtype=float)
    if bvecs.shape == (3,) and len(bvals) == 1:
        bvecs = bvecs[np.newaxis, :]
    if bvecs.shape != (len(bvals), 3):
        raise ValueError("there must be one three-dimensional b-vector per b-value")
    if not np.all(np.isfinite(bvecs)):
        raise ValueError("b-vectors must be finite")
    if np.any((bvals > 0) & (np.linalg.norm(bvecs, axis=1) == 0)):
        raise ValueError("positive b-values require nonzero b-vectors")
    if not np.isfinite(delta) or not np.isfinite(DELTA) or delta <= 0 or DELTA < delta:
        raise ValueError("delta and DELTA must be finite, with 0 < delta <= DELTA")
    if not isinstance(n_t, (int, np.integer)) or n_t < 2:
        raise ValueError("n_t must be an integer with at least two time points")
    reference_n_t = int(1e6)
    T = delta + DELTA
    reference_dt = T / (reference_n_t - 1)
    pulse_end = np.round(delta / reference_dt).astype(int)
    if pulse_end < 2:
        raise ValueError("PGSE pulses are under-resolved on the reference grid")
    if pulse_end >= reference_n_t / 2:
        # Preserve the original waveform when the pulses meet at DELTA=delta.
        gradient = np.zeros((1, reference_n_t, 3))
        gradient[0, 1:pulse_end, 0] = 1
        gradient[0, -pulse_end:-1, 0] = -1
        gradient, dt = interpolate_gradient(gradient, reference_dt, n_t)
    else:
        # The reference waveform is piecewise linear. Its eight breakpoints
        # produce the same samples as interpolating the million-point array.
        T = reference_dt * (reference_n_t - 1)
        dt = T / (n_t - 1)
        target_times = np.linspace(0, T, n_t)
        source_times = np.array(
            [
                0,
                1,
                pulse_end - 1,
                pulse_end,
                reference_n_t - pulse_end - 1,
                reference_n_t - pulse_end,
                reference_n_t - 2,
                reference_n_t - 1,
            ],
            dtype=float,
        )
        source_times *= T / (reference_n_t - 1)
        source_times[-1] = T  # np.linspace places its final point exactly at T.
        values = np.array([0, 1, 1, 0, 0, -1, -1, 0])
        gradient = np.zeros((1, n_t, 3))
        gradient[0, :, 0] = np.interp(target_times, source_times, values)
    target_times = np.linspace(0, T, n_t)
    first_pulse = (target_times > 0) & (target_times < delta)
    second_pulse = (target_times > DELTA) & (target_times < T)
    if (
        np.count_nonzero(gradient[0, first_pulse, 0]) < 2
        or np.count_nonzero(gradient[0, second_pulse, 0]) < 2
    ):
        raise ValueError("PGSE pulses are under-resolved: use more time points")
    gradient = np.repeat(gradient, len(bvals), axis=0)
    gradient = set_b(gradient, dt, bvals)
    Rs = np.repeat(np.eye(3)[np.newaxis, :, :], len(bvals), axis=0)
    for i, bvec in enumerate(bvecs):
        if bvals[i] > 0:
            Rs[i] = utils.vec2vec_rotmat(np.array([1.0, 0.0, 0.0]), bvec)
    gradient = rotate_gradient(gradient, Rs)
    return gradient, dt


def load_camino_scheme_file(path):
    """Generate a gradient array from a Camino general waveform scheme file. All
    waveforms must have the same number of steps and the same time step
    duration. For details, see
    http://camino.cs.ucl.ac.uk/index.php?n=Tutorials.GenwaveTutorial.

    Parameters
    ----------
    path : str
        Path to the Camino scheme file.

    Returns
    -------
    gradient : numpy.ndarray
        Gradient array.
    dt : float
        Duration of a time step in the gradient array.
    """
    with open(path, "r") as file:
        if file.readline().strip() != "VERSION: GRADIENT_WAVEFORM":
            raise Exception(
                "The scheme file does not start with 'VERSION: GRADIENT_WAVEFORM'"
            )
    scheme = np.loadtxt(path, skiprows=1)
    dts = scheme[:, 1]
    if len(set(dts)) != 1:
        raise Exception(
            "Not all rows of the scheme file have the same time step duration. "
            "Disimpy does not support scheme files with multiple time step durations."
        )
    dt = dts[0]
    gradient = scheme[:, 2::].reshape(len(scheme), -1, 3)
    return gradient, dt
