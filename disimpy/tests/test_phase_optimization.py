"""Shared-waveform phase compression and GPU signal reduction checks."""

import numpy as np
import pytest

from .. import simulations, substrates
from .._cuda import (
    IS_MLIR,
    _phase_error_upper_bound,
    _reduce_phases,
    _shared_waveform_basis,
    cuda,
)


def _gradient():
    waveform = np.array([0.0, 0.2, 0.5, -0.1, 0.0])
    amplitudes = np.random.default_rng(17).normal(size=(12, 3)) * 0.1
    return amplitudes[:, None, :] * waveform[None, :, None]


def test_shared_waveform_basis_and_fallback():
    gradient = _gradient()
    basis, coefficients = _shared_waveform_basis(gradient)
    assert basis.shape == (3, gradient.shape[1], 3)
    np.testing.assert_allclose(
        np.einsum("mk,ktd->mtd", coefficients, basis),
        gradient,
        rtol=0,
        atol=8 * np.finfo(float).eps * np.max(np.abs(gradient)),
    )

    # Independent measurement waveforms have full row rank and must fall back.
    changed = np.random.default_rng(8).normal(size=gradient.shape)
    fallback, coefficients = _shared_waveform_basis(changed)
    assert fallback is changed
    assert coefficients is None

    # A rotated three-axis family has more than three, but still few,
    # independent measurement waveforms.
    rng = np.random.default_rng(28)
    matrix = rng.normal(size=(32, 8)) @ rng.normal(size=(8, 15))
    gradient_rank8 = matrix.reshape(32, 5, 3)
    basis, coefficients = _shared_waveform_basis(gradient_rank8)
    assert basis.shape[0] == 8
    np.testing.assert_allclose(
        np.einsum("mk,ktd->mtd", coefficients, basis),
        gradient_rank8,
        rtol=1e-12,
        atol=1e-12,
    )
    unchanged, no_coefficients = _shared_waveform_basis(
        gradient_rank8, allow_bounded=False
    )
    assert unchanged is gradient_rank8
    assert no_coefficients is None

    gradient_f32 = gradient_rank8.astype(np.float32)
    basis_f32, coefficients_f32 = _shared_waveform_basis(gradient_f32)
    assert basis_f32.shape[0] == 8
    residual_f32 = np.max(
        np.abs(gradient_f32 - np.einsum("mk,ktd->mtd", coefficients_f32, basis_f32)),
        axis=(1, 2),
    )
    row_scale_f32 = np.max(np.abs(gradient_f32), axis=(1, 2))
    assert np.all(residual_f32 <= 8 * np.finfo(np.float32).eps * row_scale_f32)

    zero = np.zeros_like(gradient)
    basis, coefficients = _shared_waveform_basis(zero)
    assert basis.shape[0] == 1
    assert np.count_nonzero(coefficients) == 0


def test_phase_error_bound_covers_compressed_waveform():
    rng = np.random.default_rng(83)
    gradient = _gradient()
    basis, coefficients = _shared_waveform_basis(gradient)
    # Perturb a selected basis sample to emulate a lossy low-rank fit.
    basis = basis.copy()
    basis[0, 2, 1] += 2e-7
    dt = 2e-4
    gamma = 2.675e8
    limits = np.array([3e-6, 5e-6, 7e-6, 9e-6])
    bound = _phase_error_upper_bound(gradient, basis, coefficients, dt, limits, gamma)
    fitted = np.einsum("mk,ktd->mtd", coefficients, basis)
    residual = 0.5 * (
        (gradient[:, :-1] - fitted[:, :-1]) + (gradient[:, 1:] - fitted[:, 1:])
    )
    positions = rng.normal(size=(limits.size, 3))
    positions *= limits[:, None] / np.linalg.norm(positions, axis=1)[:, None]
    actual = np.abs(gamma * dt * np.einsum("mtd,td->m", residual, positions))
    assert np.all(actual <= bound * (1 + 1e-12))
    assert bound > 0


def test_phase_error_bound_zero_for_exact_basis():
    gradient = np.zeros((4, 5, 3))
    basis = gradient[:1].copy()
    coefficients = np.zeros((4, 1))
    assert (
        _phase_error_upper_bound(gradient, basis, coefficients, 1e-4, [1] * 4, 1) == 0
    )


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("factorable", [True, False])
def test_signal_reduction_matches_walker_signals(factorable):
    gradient = _gradient()
    if not factorable:
        gradient = np.random.default_rng(8).normal(size=gradient.shape) * 0.05
    options = {
        "n_walkers": 7,
        "diffusivity": 2e-9,
        "gradient": gradient,
        "dt": 1e-3,
        "substrate": substrates.free(),
        "seed": 13,
        "quiet": True,
    }
    per_walker = simulations.simulation(**options, all_signals=True)
    real_total = simulations.simulation(**options)
    np.testing.assert_allclose(
        real_total, per_walker.sum(axis=1), rtol=1e-12, atol=1e-12
    )


@pytest.mark.gpu
@pytest.mark.skipif(not IS_MLIR or not cuda.is_available(), reason="requires MLIR GPU")
def test_physical_gpu_collision_failure_diagnostics():
    options = {
        "n_walkers": 32,
        "gradient": np.zeros((1, 2, 3)),
        "dt": 1e-3,
        "seed": 123,
        "max_iter": 1,
        "quiet": True,
    }
    signals = simulations.simulation(
        **options, diffusivity=1e-30, substrate=substrates.sphere(1.0)
    )
    np.testing.assert_array_equal(signals, [32.0])
    with pytest.raises(RuntimeError, match="Collision resolution exceeded max_iter=1"):
        simulations.simulation(
            **options, diffusivity=2e-9, substrate=substrates.sphere(1e-9)
        )


@pytest.mark.gpu
@pytest.mark.skipif(not IS_MLIR or not cuda.is_available(), reason="requires MLIR GPU")
@pytest.mark.parametrize("factored", [True, False])
def test_multi_tile_gpu_reduction(factored):
    rng = np.random.default_rng(25)
    n_measurements = 12
    n_walkers = 257
    n_channels = 3 if factored else n_measurements
    phases = rng.normal(size=(n_channels, n_walkers))
    coefficients = rng.normal(size=(n_measurements, n_channels)) if factored else None
    stream = cuda.stream()
    d_phases = cuda.to_device(phases, stream=stream)
    expected_phases = coefficients @ phases if factored else phases
    for complex_signal in (False, True):
        actual = _reduce_phases(
            d_phases, coefficients, n_measurements, complex_signal, stream
        )
        expected = np.exp(1j * expected_phases).sum(axis=1)
        if not complex_signal:
            expected = expected.real
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
