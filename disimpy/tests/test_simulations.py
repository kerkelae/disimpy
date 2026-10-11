"""End-to-end simulation tests."""

import os
import pickle

import numpy as np
import numpy.testing as npt
import pytest

from .. import gradients, simulations, substrates

SEED = 123
pytestmark = pytest.mark.gpu


def test__fill_mesh():
    n_s = int(1e4)
    mesh_path = os.path.join(
        os.path.dirname(simulations.__file__), "tests", "sphere_mesh.pkl"
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    for n_sv in [
        np.array([1, 1, 1]),
        np.array([1, 5, 20]),
        np.array([10, 10, 10]),
    ]:
        for periodic in [True, False]:
            for padding in [np.zeros(3), np.zeros(3) + 1e-6]:
                substrate = substrates.mesh(
                    vertices, faces, periodic, padding=padding, n_sv=n_sv
                )
                points = simulations._fill_mesh(n_s, substrate, True, seed=SEED)
                r = (substrate.voxel_size - padding * 2) / 2
                points -= r + padding
                npt.assert_equal(np.max(np.linalg.norm(points, axis=1)) < r, True)
                npt.assert_almost_equal(np.mean(points, axis=0), np.zeros(3))
                points = simulations._fill_mesh(n_s, substrate, False, seed=SEED)
                points -= r + padding
                npt.assert_equal(np.min(np.linalg.norm(points, axis=1)) > 0.9 * r, True)
                npt.assert_almost_equal(np.mean(points, axis=0), np.zeros(3))
    return


def example_gradient():
    T = 80e-3
    gradient = np.zeros((1, 100, 3))
    gradient[0, 1:11, 0] = 1
    gradient[0, -11:-1, 0] = -1
    dt = T / (gradient.shape[1] - 1)
    return gradient, dt


def test_free_diffusion(tmp_path):

    # Signal
    n_s = int(1e5)
    n_t = int(1e3)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    bs = np.linspace(1, 2e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = substrates.free()
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals / n_s, np.exp(-bs * diffusivity), 2)

    # Walker trajectories
    n_s = int(1e4)
    n_t = int(1e2)
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = str(tmp_path / "example_traj.txt")
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, traj=traj_file
    )
    trajectories = np.loadtxt(traj_file)
    npt.assert_equal(trajectories.shape, (n_t, n_s * 3))
    trajectories = trajectories.reshape((n_t, n_s, 3))
    npt.assert_equal(np.prod(trajectories[0, :, :] == 0), 1)
    npt.assert_almost_equal(np.mean(trajectories[-1], axis=0), 0, 5)
    expected_msd = 6 * diffusivity * (n_t - 1) * dt
    npt.assert_allclose(
        np.mean(np.sum(trajectories[-1] ** 2, axis=1)),
        expected_msd,
        rtol=0.03,
    )
    return


def test_cylinder_diffusion(tmp_path):

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = str(tmp_path / "example_traj.txt")
    for radius in [1e-6, 5e-6, 1e-3]:
        substrate = substrates.cylinder(
            radius=radius, orientation=np.array([1.0, 0, 0])
        )
        signals = simulations.simulation(
            n_s, diffusivity, gradient, dt, substrate, traj=traj_file
        )
        trajectories = np.loadtxt(traj_file).reshape((n_t, n_s, 3))
        max_pos = np.max(np.linalg.norm(trajectories[..., 1::], axis=2))
        npt.assert_equal(max_pos < radius, True)
        npt.assert_almost_equal(max_pos, radius)

    # Signal compared to misst
    n_s = int(1e5)
    n_t = int(1e3)

    T = 70e-3
    gradient = np.zeros((1, 700, 3))
    gradient[0, 1:300, 0] = 1
    gradient[0, -300:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.cylinder(
        orientation=np.array([0, 0, 1.0]), radius=5e-6
    )
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)

    T = 41e-3
    gradient = np.zeros((1, 410, 3))
    gradient[0, 1:10, 0] = 1
    gradient[0, -10:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.cylinder(
        orientation=np.array([0, 0, 1.0]), radius=5e-6
    )
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_cylinder_signal_smalldelta_1ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)

    # Cylinder rotation
    n_s = int(1e5)
    n_t = int(1e3)
    gradient, dt = example_gradient()
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    substrate = substrates.cylinder(orientation=np.array([1.0, 0, 1.0]), radius=5e-6)
    signals_1 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    substrate = substrates.cylinder(orientation=-np.array([1.0, 0, 1.0]), radius=5e-6)
    signals_2 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_1 / n_s, signals_2 / n_s)
    substrate = substrates.cylinder(orientation=-np.array([1.0, 0, 0]), radius=5e-6)
    signals_3 = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals_3 / n_s, np.exp(-bs * diffusivity), 2)

    return


def test_sphere_diffusion(tmp_path):

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = str(tmp_path / "example_traj.txt")
    radius = 5e-6
    substrate = substrates.sphere(radius)
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, traj=traj_file
    )
    trajectories = np.loadtxt(traj_file).reshape((n_t, n_s, 3))
    max_pos = np.max(np.linalg.norm(trajectories, axis=2))
    npt.assert_equal(max_pos < radius, True)
    npt.assert_almost_equal(max_pos, radius)

    # Signal compared to misst
    n_s = int(1e5)
    n_t = int(1e3)

    T = 70e-3
    gradient = np.zeros((1, 700, 3))
    gradient[0, 1:300, 0] = 1
    gradient[0, -300:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.sphere(radius=5e-6)
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_sphere_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)

    T = 41e-3
    gradient = np.zeros((1, 410, 3))
    gradient[0, 1:10, 0] = 1
    gradient[0, -10:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)
    cylinder_substrate = substrates.sphere(radius=5e-6)
    signals = simulations.simulation(n_s, diffusivity, gradient, dt, cylinder_substrate)
    misst_signals_path = os.path.join(
        os.path.dirname(gradients.__file__),
        "tests",
        "misst_sphere_signal_smalldelta_1ms_bigdelta_40ms_radius_5um.txt",
    )
    misst_signals = np.loadtxt(misst_signals_path)
    npt.assert_almost_equal(signals / n_s, misst_signals, 2)
    return


def test_ellipsoid_diffusion(tmp_path):

    # Walker trajectories
    n_s = int(1e2)
    n_t = int(1e2)
    diffusivity = 2e-9
    gradient, dt = example_gradient()
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    traj_file = str(tmp_path / "example_traj.txt")
    radius = 5e-6
    semiaxes = np.ones(3) * radius
    substrate = substrates.ellipsoid(semiaxes)
    signals = simulations.simulation(
        n_s, diffusivity, gradient, dt, substrate, traj=traj_file
    )
    trajectories = np.loadtxt(traj_file).reshape((n_t, n_s, 3))
    max_pos = np.max(np.linalg.norm(trajectories, axis=2))
    npt.assert_equal(max_pos < radius, True)
    npt.assert_almost_equal(max_pos, radius)

    # Compare signal to a sphere
    substrate = substrates.sphere(radius)
    signals_sphere = simulations.simulation(n_s, diffusivity, gradient, dt, substrate)
    npt.assert_almost_equal(signals, signals_sphere)
    return


def test_mesh_diffusion():

    n_s = int(1e4)
    n_t = int(1e3)
    diffusivity = 2e-9

    mesh_path = os.path.join(
        os.path.dirname(simulations.__file__),
        "tests",
        "cylinder_mesh_closed.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]

    T = 70e-3
    gradient = np.zeros((1, 700, 3))
    gradient[0, 1:300, 0] = 1
    gradient[0, -300:-1, 0] = -1
    bs = np.linspace(1, 3e9, 100)
    gradient = np.concatenate([gradient for _ in bs], axis=0)
    dt = T / (gradient.shape[1] - 1)
    gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)
    gradient = gradients.set_b(gradient, dt, bs)

    for periodic in [True, False]:
        for padding in [np.zeros(3), np.zeros(3) + 1e-6]:
            for n_sv in [
                np.array([1, 1, 1]),
                np.array([1, 5, 20]),
                np.array([10, 10, 10]),
            ]:
                substrate = substrates.mesh(
                    vertices,
                    faces,
                    periodic,
                    padding=padding,
                    init_pos="intra",
                    n_sv=n_sv,
                )

                signals, pos = simulations.simulation(
                    n_s, diffusivity, gradient, dt, substrate, final_pos=True
                )

                # Compare to misst signal
                misst_signals_path = os.path.join(
                    os.path.dirname(gradients.__file__),
                    "tests",
                    "misst_cylinder_signal_smalldelta_30ms_bigdelta_40ms_radius_5um.txt",
                )
                misst_signals = np.loadtxt(misst_signals_path)
                npt.assert_almost_equal(signals / n_s, misst_signals, 2)

                # Make sure no spins leaked
                r = np.max(
                    np.linalg.norm(
                        substrate.vertices[:, 0:2]
                        - (substrate.voxel_size[0:2] - padding[0:2] * 2) / 2,
                        axis=1,
                    )
                )
                voxel_z = substrate.voxel_size[2]
                npt.assert_equal(np.min(pos[:, 2]) > 0, True)
                npt.assert_equal(np.max(pos[:, 2]) < voxel_z, True)
                npt.assert_equal(
                    np.max(
                        np.linalg.norm(
                            pos[:, 0:2] - np.max(substrate.vertices, axis=0)[0:2] / 2,
                            axis=1,
                        )
                    )
                    < r,
                    True,
                )

    # Test with a periodic mesh
    mesh_path = os.path.join(
        os.path.dirname(simulations.__file__),
        "tests",
        "cylinder_mesh_open.pkl",
    )
    with open(mesh_path, "rb") as f:
        example_mesh = pickle.load(f)
    faces = example_mesh["faces"]
    vertices = example_mesh["vertices"]
    init_pos = np.zeros((n_s, 3)) + np.array([5e-6, 5e-6, 12.5e-6])

    for padding in [np.zeros(3), np.array([1e-6, 1e-6, 0])]:
        for n_sv in [
            np.array([1, 1, 1]),
            np.array([1, 5, 20]),
            np.array([10, 10, 10]),
        ]:
            substrate = substrates.mesh(
                vertices,
                faces,
                init_pos=init_pos + padding,
                periodic=True,
                padding=padding,
                n_sv=n_sv,
            )
            signals, pos = simulations.simulation(
                n_s, diffusivity, gradient, dt, substrate, final_pos=True
            )
            r = np.max(
                np.linalg.norm(
                    substrate.vertices[:, 0:2]
                    - (substrate.voxel_size[0:2] - padding[0:2] * 2) / 2,
                    axis=1,
                )
            )
            voxel_z = substrate.voxel_size[2]
            npt.assert_equal(np.min(pos[:, 2]) < 0, True)
            npt.assert_equal(np.max(pos[:, 2]) > voxel_z, True)
            npt.assert_equal(
                np.max(
                    np.linalg.norm(
                        pos[:, 0:2] - np.max(substrate.vertices, axis=0)[0:2] / 2,
                        axis=1,
                    )
                )
                < r,
                True,
            )

    # Test with neuron model and confirm that no spins leak
    n_s = int(1e3)
    n_t = int(1e2)
    gradient = np.ones((1, n_t, 3))
    for dt in [1e-5, 1e-3, 1e-1]:
        mesh_path = os.path.join(
            os.path.dirname(simulations.__file__), "tests", "neuron-model.pkl"
        )
        with open(mesh_path, "rb") as f:
            example_mesh = pickle.load(f)
        faces = example_mesh["faces"]
        vertices = example_mesh["vertices"]
        substrate = substrates.mesh(vertices, faces, init_pos="intra", periodic=True)
        signals, pos = simulations.simulation(
            n_s, diffusivity, gradient, dt, substrate, final_pos=True
        )
        npt.assert_equal(np.all(np.max(pos, axis=0) < substrate.voxel_size), True)
        npt.assert_equal(np.all(np.min(pos, axis=0) > 0), True)
    return
