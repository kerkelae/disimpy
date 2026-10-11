"""Structural checks for CUDA phase-reduction launch limits."""

from types import SimpleNamespace

import numpy as np
import pytest

from .. import _cuda
from .._cuda import IS_MLIR, cuda


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires CUDA")
def test_flattened_phase_grid_matches_two_tile_reference():
    rng = np.random.default_rng(741)
    phases = rng.normal(size=(3, 257))
    stream = cuda.stream()
    actual = _cuda._reduce_phases(
        cuda.to_device(phases, stream=stream),
        coefficients=None,
        n_measurements=phases.shape[0],
        complex_signal=True,
        stream=stream,
    )

    np.testing.assert_allclose(
        actual,
        np.exp(1j * phases).sum(axis=1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_phase_partial_grid_flattens_measurements_beyond_grid_y(monkeypatch):
    """65,536 measurements must use grid x without materializing partials."""

    class ShapeOnlyArray:
        def __init__(self, shape):
            self.shape = shape if isinstance(shape, tuple) else (shape,)

        def copy_to_host(self, stream=None):
            return np.zeros(self.shape, dtype=np.float64)

    class CaptureKernel:
        def __init__(self):
            self.configurations = []

        def __getitem__(self, configuration):
            self.configurations.append(configuration)
            return lambda *args: None

    class Stream:
        def synchronize(self):
            return None

    partials_kernel = CaptureKernel()
    finish_kernel = CaptureKernel()
    fake_cuda = SimpleNamespace(
        device_array=lambda shape, dtype, stream=None: ShapeOnlyArray(shape),
        to_device=lambda array, stream=None: ShapeOnlyArray(array.shape),
    )
    monkeypatch.setattr(_cuda, "cuda", fake_cuda)
    monkeypatch.setattr(_cuda, "_cuda_phase_partials", partials_kernel)
    monkeypatch.setattr(_cuda, "_cuda_phase_finish", finish_kernel)

    n_measurements = 65_536
    result = _cuda._reduce_phases(
        ShapeOnlyArray((1, 1)),
        coefficients=None,
        n_measurements=n_measurements,
        complex_signal=False,
        stream=Stream(),
    )

    partial_grid, partial_threads, _ = partials_kernel.configurations[0]
    assert partial_grid == n_measurements
    assert isinstance(partial_grid, int)
    assert partial_threads == 1
    assert finish_kernel.configurations[0][0] == n_measurements
    assert result.shape == (n_measurements,)
