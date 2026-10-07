"""Small simulations for CPU execution through the CUDA simulator."""

import os

import numpy as np
import pytest

from .. import simulations, substrates


pytestmark = pytest.mark.skipif(
    os.environ.get("NUMBA_ENABLE_CUDASIM") != "1",
    reason="requires NUMBA_ENABLE_CUDASIM=1 before import",
)


def test_free_diffusion():
    gradient = np.zeros((2, 4, 3), dtype=np.float64)
    gradient[1, 1:3, 0] = 0.5
    options = dict(
        n_walkers=4,
        diffusivity=2e-9,
        gradient=gradient,
        dt=1e-3,
        substrate=substrates.free(),
        seed=123,
        final_pos=True,
        all_signals=True,
        quiet=True,
        cuda_bs=4,
    )

    signals, positions = simulations.simulation(**options)
    repeated_signals, repeated_positions = simulations.simulation(**options)

    assert signals.shape == (2, 4)
    assert positions.shape == (4, 3)
    np.testing.assert_array_equal(signals[0], np.ones(4))
    assert np.all(np.isfinite(signals))
    assert np.all(np.abs(signals) <= 1)
    assert np.any(signals[1] < 1)
    np.testing.assert_array_equal(signals, repeated_signals)
    np.testing.assert_array_equal(positions, repeated_positions)
