"""Direct GPU initial positions stay in the requested physical domains."""

import numpy as np
import pytest

from .._cuda import IS_MLIR, _device_initial_positions, cuda


@pytest.mark.skipif(IS_MLIR and not cuda.is_available(), reason="requires CUDA GPU")
@pytest.mark.parametrize("kind", ["sphere", "cylinder", "ellipsoid", "uniform_voxel"])
def test_device_initial_positions_are_seeded_and_within_domain(kind):
    stream = cuda.stream()
    radius = 1.5
    dimensions = np.array([1.0, 1.5, 2.0])
    d_dimensions = cuda.to_device(dimensions, stream=stream)
    options = {
        "radius": radius if kind in ("sphere", "cylinder") else None,
        "dimensions": d_dimensions if kind in ("ellipsoid", "uniform_voxel") else None,
    }
    first = _device_initial_positions(kind, 8, 37, 8, stream, **options)
    second = _device_initial_positions(kind, 8, 37, 8, stream, **options)
    positions = first.copy_to_host(stream=stream)
    np.testing.assert_array_equal(positions, second.copy_to_host(stream=stream))
    stream.synchronize()
    if kind == "sphere":
        assert np.all(np.linalg.norm(positions, axis=1) <= radius)
    elif kind == "cylinder":
        np.testing.assert_array_equal(positions[:, 0], 0.0)
        assert np.all(np.linalg.norm(positions[:, 1:], axis=1) <= radius)
    elif kind == "ellipsoid":
        assert np.all(np.linalg.norm(positions / dimensions, axis=1) <= 1.0)
    else:
        assert np.all(positions >= 0.0)
        assert np.all(positions <= dimensions)


@pytest.mark.gpu
@pytest.mark.skipif(not IS_MLIR or not cuda.is_available(), reason="requires MLIR GPU")
@pytest.mark.parametrize("kind", ["sphere", "cylinder", "ellipsoid", "uniform_voxel"])
def test_device_initial_positions_have_uniform_radial_or_axis_moments(kind):
    stream = cuda.stream()
    dimensions = np.array([1.0, 1.5, 2.0])
    d_dimensions = cuda.to_device(dimensions, stream=stream)
    points = _device_initial_positions(
        kind,
        32768,
        37,
        128,
        stream,
        radius=1.5 if kind in ("sphere", "cylinder") else None,
        dimensions=d_dimensions if kind in ("ellipsoid", "uniform_voxel") else None,
    )
    positions = points.copy_to_host(stream=stream)
    stream.synchronize()
    if kind == "sphere":
        volume_fraction = (np.linalg.norm(positions, axis=1) / 1.5) ** 3
        np.testing.assert_allclose(volume_fraction.mean(), 0.5, atol=0.015)
        np.testing.assert_allclose(positions.mean(axis=0), 0.0, atol=0.02)
    elif kind == "cylinder":
        area_fraction = (np.linalg.norm(positions[:, 1:], axis=1) / 1.5) ** 2
        np.testing.assert_allclose(area_fraction.mean(), 0.5, atol=0.015)
        np.testing.assert_allclose(positions[:, 1:].mean(axis=0), 0.0, atol=0.02)
    elif kind == "ellipsoid":
        volume_fraction = np.linalg.norm(positions / dimensions, axis=1) ** 3
        np.testing.assert_allclose(volume_fraction.mean(), 0.5, atol=0.015)
        np.testing.assert_allclose(
            (positions / dimensions).mean(axis=0), 0.0, atol=0.02
        )
    else:
        np.testing.assert_allclose(
            (positions / dimensions).mean(axis=0), 0.5, atol=0.015
        )
