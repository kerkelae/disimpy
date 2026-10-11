"""CPU-only regression tests for simulation helpers and input validation."""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from .. import _cuda, simulations, substrates


@pytest.mark.parametrize("seed", [0, 17])
def test_add_noise_seed_repeats_reference_values(seed):
    data = np.array([0.0, 0.25, 1.0, 2.0])
    sigma = 0.2
    random = np.random.RandomState(seed)
    expected = np.abs(
        data
        + random.normal(size=data.shape, scale=sigma, loc=0)
        + 1j * random.normal(size=data.shape, scale=sigma, loc=0)
    )

    original_state = np.random.get_state()
    try:
        np.random.seed(741)
        first = simulations.add_noise_to_data(data, sigma, seed=seed)
        np.random.seed(231)
        second = simulations.add_noise_to_data(data, sigma, seed=seed)
    finally:
        np.random.set_state(original_state)

    npt.assert_array_equal(first, expected)
    npt.assert_array_equal(second, expected)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (
            np.float64,
            b"0.0 -0.0 1.23456789 1e-07 1e+20 inf nan -1e-20 16777217.0 \n",
        ),
        (
            np.float32,
            b"0.0 -0.0 1.2345679 1e-07 1e+20 inf nan -1e-20 1.6777216e+07 \n",
        ),
    ],
    ids=["float64", "float32"],
)
def test_write_traj_preserves_scalar_text(tmp_path, dtype, expected):
    positions = np.array(
        [[0.0, -0.0, 1.23456789], [1e-7, 1e20, np.inf], [np.nan, -1e-20, 16777217.0]],
        dtype=dtype,
    )
    trajectory = tmp_path / "trajectory.txt"

    with trajectory.open("w") as file:
        _cuda._write_traj_row(file, positions)
    assert trajectory.read_bytes() == expected


def test_invalid_n_walkers_does_not_probe_gpu(monkeypatch):
    detect_calls = []

    def detect():
        detect_calls.append(True)
        raise RuntimeError("GPU detection should not run for invalid input")

    monkeypatch.setattr(simulations.cuda, "detect", detect)

    with pytest.raises(ValueError, match=r"Incorrect value \(0\) for n_walkers"):
        simulations.simulation(
            n_walkers=0,
            diffusivity=2e-9,
            gradient=np.zeros((1, 2, 3), dtype=np.float64),
            dt=1e-3,
            substrate=substrates.free(),
            quiet=True,
        )

    assert detect_calls == []


def test_gpu_detection_error_keeps_cause_and_setup_link(monkeypatch):
    original_error = RuntimeError("driver unavailable")

    def detect():
        raise original_error

    monkeypatch.setattr(simulations.cuda, "detect", detect)

    with pytest.raises(RuntimeError, match="Unable to detect a CUDA GPU") as error:
        simulations.simulation(
            n_walkers=1,
            diffusivity=2e-9,
            gradient=np.zeros((1, 2, 3), dtype=np.float64),
            dt=1e-3,
            substrate=substrates.free(),
            quiet=True,
        )

    assert (
        "https://nvidia.github.io/numba-cuda-mlir/latest/user/installation.html"
        in str(error.value)
    )
    assert error.value.__cause__ is original_error


def test_gpu_detection_false_is_failure(monkeypatch):
    monkeypatch.setattr(simulations.cuda, "detect", lambda: False)

    with pytest.raises(RuntimeError, match="Unable to detect a CUDA GPU"):
        simulations.simulation(
            n_walkers=1,
            diffusivity=2e-9,
            gradient=np.zeros((1, 2, 3), dtype=np.float64),
            dt=1e-3,
            substrate=substrates.free(),
            quiet=True,
        )


def test_seed_above_uint64_range_fails_before_gpu_detection(monkeypatch):
    detect_calls = []
    monkeypatch.setattr(simulations.cuda, "detect", lambda: detect_calls.append(True))

    with pytest.raises(ValueError, match=r"Incorrect value \(18446744073709551616\)"):
        simulations.simulation(
            n_walkers=1,
            diffusivity=2e-9,
            gradient=np.zeros((1, 2, 3), dtype=np.float64),
            dt=1e-3,
            substrate=substrates.free(),
            seed=1 << 64,
            quiet=True,
        )

    assert detect_calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"diffusivity": float("nan")}, "diffusivity"),
        ({"diffusivity": float("inf")}, "diffusivity"),
        ({"dt": float("nan")}, "dt"),
        ({"dt": float("inf")}, "dt"),
        ({"gradient": np.full((1, 2, 3), np.nan)}, "gradient"),
        ({"gradient": np.empty((0, 2, 3))}, "gradient"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"epsilon": float("nan")}, "epsilon"),
        ({"epsilon": float("inf")}, "epsilon"),
    ],
)
def test_nonfinite_and_unsuitable_simulation_inputs_fail_before_gpu(
    monkeypatch, change, message
):
    detect_calls = []

    def detect():
        detect_calls.append(True)

    monkeypatch.setattr(simulations.cuda, "detect", detect)
    options = {
        "n_walkers": 1,
        "diffusivity": 2e-9,
        "gradient": np.zeros((1, 2, 3), dtype=np.float64),
        "dt": 1e-3,
        "substrate": substrates.free(),
        "quiet": True,
    }
    options.update(change)

    with pytest.raises(ValueError, match=message):
        simulations.simulation(**options)

    assert detect_calls == []


@pytest.mark.parametrize(
    ("substrate", "epsilon"),
    [
        (substrates.sphere(2e-6), 2e-6),
        (substrates.cylinder(3e-6, np.array([1.0, 0.0, 0.0])), 4e-6),
        (substrates.ellipsoid(np.array([2e-6, 3e-6, 4e-6])), 2e-6),
    ],
)
def test_epsilon_must_be_smaller_than_restricted_geometry_scale(
    monkeypatch, substrate, epsilon
):
    detect_calls = []
    monkeypatch.setattr(simulations.cuda, "detect", lambda: detect_calls.append(True))

    with pytest.raises(ValueError, match="epsilon.*substrate length scale"):
        simulations.simulation(
            n_walkers=1,
            diffusivity=2e-9,
            gradient=np.zeros((1, 2, 3), dtype=np.float64),
            dt=1e-3,
            substrate=substrate,
            epsilon=epsilon,
            quiet=True,
        )

    assert detect_calls == []


def test_fill_mesh_has_bounded_failure_with_diagnostic(monkeypatch):
    class DeviceArray:
        def __init__(self, array):
            self.array = np.array(array, copy=True)

        def copy_to_host(self, stream=None):
            return self.array.copy()

        def copy_to_device(self, array, stream=None):
            self.array[...] = array

    class Stream:
        def synchronize(self):
            return None

    class RejectAllKernel:
        launches = 0
        fail_classification = False

        def __getitem__(self, configuration):
            def launch(*args):
                self.launches += 1
                args[1].array.fill(0)
                args[2].array.fill(int(self.fail_classification))

            return launch

    class NoOpKernel:
        def __getitem__(self, configuration):
            return lambda *args: None

    class PrefixKernel:
        def __getitem__(self, configuration):
            def launch(block_counts, block_offsets, batch_accepted):
                block_offsets.array.fill(0)
                batch_accepted.array[0] = int(np.sum(block_counts.array))

            return launch

    fake_cuda = SimpleNamespace(
        stream=Stream,
        to_device=lambda array, stream=None: DeviceArray(array),
        device_array=lambda shape, dtype, stream=None: DeviceArray(
            np.empty(shape, dtype=dtype)
        ),
    )
    kernel = RejectAllKernel()
    monkeypatch.setattr(simulations, "cuda", fake_cuda)
    monkeypatch.setattr(simulations, "_cuda_fill_mesh", kernel)
    monkeypatch.setattr(simulations, "_cuda_fill_mesh_cover", kernel)
    monkeypatch.setattr(simulations, "_cuda_prefix_mesh_block_counts", PrefixKernel())
    monkeypatch.setattr(simulations, "_cuda_compact_mesh_candidates", NoOpKernel())
    monkeypatch.setattr(simulations, "_cuda_copy_rng_states", NoOpKernel())
    monkeypatch.setattr(
        simulations,
        "create_xoroshiro128p_states",
        lambda *args, **kwargs: object(),
    )
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-6, 0.0, 0.0],
            [0.0, 1e-6, 0.0],
            [0.0, 0.0, 1e-6],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    substrate = substrates.mesh(
        vertices,
        faces,
        periodic=True,
        init_pos="intra",
        n_sv=np.ones(3, dtype=int),
        quiet=True,
    )

    with pytest.raises(
        RuntimeError,
        match="after 3 candidate batches; accepted 0",
    ):
        simulations._fill_mesh(
            2, substrate, intra=True, seed=123, cuda_bs=2, max_batches=3
        )

    assert kernel.launches == 3

    kernel.launches = 0
    kernel.fail_classification = True
    with pytest.raises(RuntimeError, match="classification failed.*batch 1"):
        simulations._fill_mesh(
            2, substrate, intra=True, seed=123, cuda_bs=2, max_batches=3
        )
    assert kernel.launches == 1
    kernel.fail_classification = False

    kernel.launches = 0
    with pytest.raises(RuntimeError, match="accepted 0"):
        simulations._fill_mesh(
            1000, substrate, intra=True, seed=123, cuda_bs=128, max_batches=1000
        )
    assert kernel.launches < 100
