"""This module contains code for running diffusion-weighted MR simulations."""

import contextlib
import math
import os
import sys

import numpy as np

from . import substrates, utils
from ._cuda import (
    _cuda_compact_mesh_candidates,
    _cuda_copy_rng_states,
    _cuda_fill_mesh,
    _cuda_fill_mesh_cover,
    _cuda_fused_free_large_phase,
    _cuda_fused_free_small_phase,
    _cuda_fused_local_analytic_small_phase,
    _cuda_fused_mesh_small_phase,
    _cuda_fused_sphere_small_phase,
    _cuda_prefix_mesh_block_counts,
    _cuda_step_cylinder_local,
    _cuda_step_ellipsoid_local,
    _cuda_step_free,
    _cuda_step_mesh,
    _cuda_step_sphere,
    _device_initial_positions,
    _phase_error_upper_bound,
    _reduce_phases,
    _shared_waveform_basis,
    _TrajectoryRecorder,
    create_local_xoroshiro128p_states,
    create_xoroshiro128p_states,
    cuda,
    derive_seed,
    empty_xoroshiro128p_states,
)
from ._mesh import _classification_mesh_lookup, _mesh_sampling_grid
from .gradients import GAMMA, _validate_dt, _validate_gradient

_MAX_CUDA_THREADS_PER_BLOCK = 1024
_MAX_PHASE_COMPRESSION_ERROR_RAD = 1e-6


def _step_chunk_size(kind, n_walkers, n_steps, n_channels):
    """Use launch fusion in the physical GPU regimes where it won.

    RTX A1000 paired runs across 512–16384 walkers and 63–511 steps favored
    chunks of 32–64 steps. Shorter runs stay on the one-step path pending
    direct evidence. The private selector permits paired benchmark overrides.
    """
    if n_steps < 16:
        return 1
    if kind != "free" and n_channels > 16:
        return 1
    return 32 if n_walkers >= 8192 else 64


def _position_radius_bounds(substrate, n_steps, step_l, max_iter, epsilon):
    """Bound positions from the global origin during phase accumulation."""
    steps = np.arange(1, n_steps + 1, dtype=np.float64)
    maximal_step = step_l + max_iter * epsilon
    if substrate.type == "free":
        return steps * step_l
    if substrate.type == "sphere":
        return np.full(n_steps, substrate.radius + max_iter * epsilon)
    if substrate.type == "cylinder":
        return substrate.radius + steps * maximal_step
    if substrate.type == "ellipsoid":
        return np.full(n_steps, np.max(substrate.semiaxes) + max_iter * epsilon)
    if substrate.type == "mesh":
        return np.linalg.norm(substrate.voxel_size) + steps * maximal_step
    return steps * maximal_step


def _validate_mesh_initial_positions(substrate, n_walkers):
    """Recheck mutable initial positions before allocating GPU resources."""
    initial = substrate.init_pos
    if isinstance(initial, np.ndarray):
        if (
            initial.shape != (n_walkers, 3)
            or not np.issubdtype(initial.dtype, np.floating)
            or not np.all(np.isfinite(initial))
            or np.any(initial < 0)
            or np.any(initial > substrate.voxel_size)
        ):
            raise ValueError("Invalid mesh init_pos array for n_walkers and voxel")
    elif not isinstance(initial, str) or initial not in ("uniform", "intra", "extra"):
        raise ValueError(
            "Mesh init_pos must be an array, 'uniform', 'intra', or 'extra'"
        )


def _fill_mesh(
    n_points,
    substrate,
    intra,
    seed,
    cuda_bs=128,
    max_batches=1000,
    *,
    stream=None,
    device_mesh=None,
    return_device=False,
):
    """Sample points from a uniform distribution inside or outside the surface
    defined by a triangular mesh.

    Parameters
    ----------
    n_points : int
        Number of positions to return.
    substrate : substrates._Substrate
        Validated mesh substrate.
    intra : bool
        If True, sample inside the mesh; otherwise sample outside it.
    seed : int
        Random seed.
    cuda_bs : int, optional
    max_batches : int, optional
        Maximum number of candidate/compaction batches before sampling is
        aborted.

    Returns
    -------
    points : np.ndarray
    """
    if not isinstance(max_batches, int) or max_batches < 1:
        raise ValueError(f"Incorrect value ({max_batches}) for max_batches")
    if (
        not isinstance(cuda_bs, int)
        or cuda_bs < 1
        or cuda_bs > _MAX_CUDA_THREADS_PER_BLOCK
    ):
        raise ValueError(f"Incorrect value ({cuda_bs}) for cuda_bs")
    sampling_grid = _mesh_sampling_grid(substrate, intra=intra)
    bs = cuda_bs
    batch_size = n_points
    gs = (batch_size + bs - 1) // bs
    if stream is None:
        stream = cuda.stream()
    # Mesh rejection sampling must not reuse the diffusion stream's first
    # values: that correlates accepted positions with the walkers' first step.
    rng_states = create_xoroshiro128p_states(
        gs * bs, seed=derive_seed(seed, 1), stream=stream
    )

    if device_mesh is None:
        device_mesh = _upload_mesh(substrate, stream, classification=True)
    if sampling_grid is not None:
        cell_ids, cell_shape, cell_size = sampling_grid
        if cell_ids.size == 0:
            raise RuntimeError("Mesh sampling grid contains no candidate cells")
        d_cell_ids = cuda.to_device(cell_ids, stream=stream)
        d_cell_shape = cuda.to_device(cell_shape, stream=stream)
        d_cell_size = cuda.to_device(cell_size, stream=stream)
    d_points = cuda.device_array((n_points, 3), dtype=np.float64, stream=stream)
    d_candidates = cuda.device_array((batch_size, 3), dtype=np.float64, stream=stream)
    d_block_counts = cuda.device_array(gs, dtype=np.int64, stream=stream)
    d_block_offsets = cuda.device_array(gs, dtype=np.int64, stream=stream)
    batch_accepted = np.zeros(1, dtype=np.int64)
    d_batch_accepted = cuda.to_device(batch_accepted, stream=stream)
    classifier_failure = np.zeros(1, dtype=np.int64)
    d_classifier_failure = cuda.to_device(classifier_failure, stream=stream)
    accepted = 0
    total_candidates = 0
    # Sparse compartments need more candidates per launch. The cap bounds
    # temporary GPU storage even when the requested walker count is large.
    max_batch_size = max(batch_size, min(65536, batch_size * 64))
    for batch in range(max_batches):
        classifier_failure[0] = 0
        d_classifier_failure.copy_to_device(classifier_failure, stream=stream)
        if sampling_grid is None:
            _cuda_fill_mesh[gs, bs, stream](
                d_candidates,
                d_block_counts,
                d_classifier_failure,
                rng_states,
                intra,
                device_mesh["triangle_data"],
                device_mesh["voxel_size"],
                device_mesh["classification_triangle_indices"],
                device_mesh["classification_subvoxel_indices"],
                device_mesh["xs"],
                device_mesh["ys"],
                device_mesh["zs"],
                device_mesh["n_sv"],
                substrate.n_membrane_faces,
            )
        else:
            _cuda_fill_mesh_cover[gs, bs, stream](
                d_candidates,
                d_block_counts,
                d_classifier_failure,
                rng_states,
                intra,
                d_cell_ids,
                d_cell_shape,
                d_cell_size,
                device_mesh["triangle_data"],
                device_mesh["voxel_size"],
                device_mesh["classification_triangle_indices"],
                device_mesh["classification_subvoxel_indices"],
                device_mesh["xs"],
                device_mesh["ys"],
                device_mesh["zs"],
                device_mesh["n_sv"],
                substrate.n_membrane_faces,
            )
        _cuda_prefix_mesh_block_counts[1, 1, stream](
            d_block_counts, d_block_offsets, d_batch_accepted
        )
        batch_accepted = d_batch_accepted.copy_to_host(stream=stream)
        classifier_failure = d_classifier_failure.copy_to_host(stream=stream)
        stream.synchronize()
        if classifier_failure[0]:
            raise RuntimeError(
                "Mesh initial-position classification failed: ordered ray "
                "classification exceeded its geometry-derived crossing bound "
                f"for {int(classifier_failure[0])} candidate(s) in batch "
                f"{batch + 1}."
            )
        _cuda_compact_mesh_candidates[gs, bs, stream](
            d_candidates,
            d_block_counts,
            d_block_offsets,
            accepted,
            d_points,
        )
        accepted += int(batch_accepted[0])
        total_candidates += batch_size
        if accepted >= n_points:
            if return_device:
                return d_points
            points = d_points.copy_to_host(stream=stream)
            stream.synchronize()
            return points

        if (batch + 1) % 8 == 0 and accepted * 16 < total_candidates:
            if batch_size < max_batch_size:
                new_batch_size = min(batch_size * 4, max_batch_size)
                new_gs = (new_batch_size + bs - 1) // bs
                new_rng_states = create_xoroshiro128p_states(
                    new_gs * bs, seed=derive_seed(seed, 1), stream=stream
                )
                # Keep the advanced streams of old lanes. Newly activated
                # lanes retain their independently seeded stream indices.
                _cuda_copy_rng_states[gs, bs, stream](rng_states, new_rng_states)
                rng_states = new_rng_states
                batch_size = new_batch_size
                gs = new_gs
                d_candidates = cuda.device_array(
                    (batch_size, 3), dtype=np.float64, stream=stream
                )
                d_block_counts = cuda.device_array(gs, dtype=np.int64, stream=stream)
                d_block_offsets = cuda.device_array(gs, dtype=np.int64, stream=stream)

        if accepted == 0:
            # After zero hits, use an exact one-sided 1e-12 confidence bound
            # for the acceptance rate. Stop only when a further Chernoff
            # bound puts the chance of filling the remaining slots below
            # 1e-6, even at that upper rate.
            remaining_capacity = (max_batches - batch - 1) * max_batch_size
            if total_candidates and remaining_capacity:
                rate_upper = -math.expm1(math.log(1e-12) / total_candidates)
                mean_upper = rate_upper * remaining_capacity
                if 0 < mean_upper < n_points:
                    log_success_bound = (
                        -n_points * math.log(n_points / mean_upper)
                        + n_points
                        - mean_upper
                    )
                    if log_success_bound < math.log(1e-6):
                        break

    compartment = "inside" if intra else "outside"
    raise RuntimeError(
        f"Could not sample {n_points} mesh initial position(s) {compartment} "
        f"the surface after {batch + 1} candidate batches; accepted "
        f"{min(accepted, n_points)}. The requested compartment may have zero or "
        f"very small volume."
    )


def _upload_mesh(substrate, stream, classification=False):
    """Upload one mesh once for initialization and collision simulation."""
    triangle_indices = substrate.triangle_indices
    subvoxel_indices = substrate.subvoxel_indices
    d_triangle_indices = cuda.to_device(triangle_indices, stream=stream)
    d_subvoxel_indices = cuda.to_device(subvoxel_indices, stream=stream)
    d_classification_triangle_indices = None
    d_classification_subvoxel_indices = None
    if classification:
        classification_triangles, classification_subvoxels = (
            _classification_mesh_lookup(substrate)
        )
        if classification_triangles.size == triangle_indices.size:
            d_classification_triangle_indices = d_triangle_indices
            d_classification_subvoxel_indices = d_subvoxel_indices
        else:
            d_classification_triangle_indices = cuda.to_device(
                classification_triangles, stream=stream
            )
            d_classification_subvoxel_indices = cuda.to_device(
                classification_subvoxels, stream=stream
            )
    return {
        "triangle_data": cuda.to_device(substrate.triangle_data, stream=stream),
        "voxel_size": cuda.to_device(substrate.voxel_size, stream=stream),
        "xs": cuda.to_device(substrate.xs, stream=stream),
        "ys": cuda.to_device(substrate.ys, stream=stream),
        "zs": cuda.to_device(substrate.zs, stream=stream),
        "triangle_indices": d_triangle_indices,
        "subvoxel_indices": d_subvoxel_indices,
        "classification_triangle_indices": d_classification_triangle_indices,
        "classification_subvoxel_indices": d_classification_subvoxel_indices,
        "n_sv": cuda.to_device(substrate.n_sv, stream=stream),
    }


def add_noise_to_data(data, sigma, seed=None):
    """Add Rician noise to data.

    Parameters
    ----------
    data : numpy.ndarray
        Array containing the data.
    sigma : float
        Standard deviation of noise in each channel.
    seed : int, optional
        Seed for pseudorandom number generation.

    Returns
    -------
    noisy_data : numpy.ndarray
        Noisy data.
    """
    if seed is not None:
        np.random.seed(seed)
    noisy_data = np.abs(
        data
        + np.random.normal(size=data.shape, scale=sigma, loc=0)
        + 1j * np.random.normal(size=data.shape, scale=sigma, loc=0)
    )
    return noisy_data


def simulation(
    n_walkers,
    diffusivity,
    gradient,
    dt,
    substrate,
    seed=123,
    traj=None,
    final_pos=False,
    all_signals=False,
    quiet=False,
    cuda_bs=128,
    max_iter=int(1e3),
    epsilon=1e-13,
):
    """Simulate a diffusion-weighted MR experiment and generate signal. For a
    detailed tutorial, please see
    https://disimpy.readthedocs.io/en/latest/tutorial.html.

    Parameters
    ----------
    n_walkers : int
        Number of random walkers.
    diffusivity : float
        Diffusivity in SI units (m^2/s).
    gradient : numpy.ndarray
        Floating-point array of shape (number of measurements, number of time
        points, 3). Array elements represent the gradient magnitude at a time
        point along an axis in SI units (T/m). ``N`` samples define ``N - 1``
        diffusion steps and a total duration of ``(N - 1) * dt``. The interval
        gradient is the average of its endpoint samples.
    dt : float
        Time between adjacent gradient samples in SI units (s).
    substrate : disimpy.substrates._Substrate
        Substrate object containing information about the simulated
        microstructure.
    seed : int, optional
        Seed for pseudorandom number generation.
    traj : str, optional
        Path of a file in which to save the simulated random walker
        trajectories. The file can become very large! Every line represents a
        gradient sample time, including the initial position. Every line
        contains the positions as follows: walker_1_x
        walker_1_y walker_1_z walker_2_x walker_2_y walker_2_z…
    final_pos : bool, optional
        If True, the signal and the final positions of the random walkers are
        returned as a tuple.
    all_signals : bool, optional
        If True, the signal from each random walker is returned instead of the
        total signal.
    quiet : bool, optional
        If True, updates on the progress of the simulation are not printed.
    cuda_bs : int, optional
        The size of the one-dimensional CUDA thread block.
    max_iter : int, optional
        The maximum number of iterations allowed in the algorithm that checks
        if a random walker collides with a surface during a time step.
    epsilon : float, optional
        The amount by which a random walker is moved away from the surface
        after a collision to avoid placing it in the surface.

    Returns
    -------
    signal : numpy.ndarray
        Simulated signals.
    """

    # Validate input
    if not isinstance(n_walkers, int) or n_walkers <= 0:
        raise ValueError("Incorrect value (%s) for n_walkers" % n_walkers)
    if (
        not isinstance(diffusivity, float)
        or not np.isfinite(diffusivity)
        or diffusivity <= 0
    ):
        raise ValueError("Incorrect value (%s) for diffusivity" % diffusivity)
    _validate_gradient(gradient)
    _validate_dt(dt)
    if not isinstance(substrate, substrates._Substrate):
        raise ValueError("Incorrect value (%s) for substrate" % substrate)
    if not isinstance(seed, int) or seed < 0 or seed > np.iinfo(np.uint64).max:
        raise ValueError("Incorrect value (%s) for seed" % seed)
    if traj is not None and not isinstance(traj, str):
        raise ValueError("Incorrect value (%s) for traj" % traj)
    if traj == "":
        raise ValueError("traj must be a nonempty path")
    if not isinstance(final_pos, bool):
        raise ValueError("Incorrect value (%s) for final_pos" % final_pos)
    if not isinstance(all_signals, bool):
        raise ValueError("Incorrect value (%s) for all_signals" % all_signals)
    if not isinstance(quiet, bool):
        raise ValueError("Incorrect value (%s) for quiet" % quiet)
    if (
        not isinstance(cuda_bs, int)
        or cuda_bs <= 0
        or cuda_bs > _MAX_CUDA_THREADS_PER_BLOCK
    ):
        raise ValueError("Incorrect value (%s) for cuda_bs" % cuda_bs)
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("Incorrect value (%s) for max_iter" % max_iter)
    if not isinstance(epsilon, float) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Incorrect value (%s) for epsilon" % epsilon)
    if substrate.type == "mesh":
        _validate_mesh_initial_positions(substrate, n_walkers)
    geometry_scale = None
    if substrate.type in ("sphere", "cylinder"):
        geometry_scale = substrate.radius
    elif substrate.type == "ellipsoid":
        geometry_scale = float(np.min(substrate.semiaxes))
    elif substrate.type == "mesh":
        # The voxel size is the only universally available mesh length scale.
        # Local gaps and triangle features can be much smaller and are checked
        # through the documented convergence study instead.
        geometry_scale = float(np.min(substrate.voxel_size))
    if geometry_scale is not None and epsilon >= geometry_scale:
        raise ValueError(
            f"epsilon ({epsilon}) must be smaller than the substrate length "
            f"scale ({geometry_scale})"
        )

    # Confirm that the CUDA backend detects a GPU without printing its details.
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        try:
            cuda_detected = cuda.detect()
        except Exception as exc:
            raise RuntimeError(
                "Unable to detect a CUDA GPU. Check that the CUDA requirements "
                "are met and the installation is configured correctly: "
                "https://nvidia.github.io/numba-cuda-mlir/latest/user/installation.html"
            ) from exc
    if cuda_detected is False:
        raise RuntimeError(
            "Unable to detect a CUDA GPU. Check that the CUDA requirements "
            "are met and the installation is configured correctly: "
            "https://nvidia.github.io/numba-cuda-mlir/latest/user/installation.html"
        )

    if not quiet:
        print("Starting simulation")
        if traj is not None:
            print(
                "The trajectories file will be up to %s GB"
                % (gradient.shape[1] * n_walkers * 3 * 25 / 1e9)
            )

    # Set up CUDA stream
    bs = cuda_bs  # Threads per block
    gs = int(math.ceil(float(n_walkers) / bs))  # Blocks per grid
    stream = cuda.stream()

    # Diffusion and position initialization use independent RNG domains.
    # Fused kernels seed their local diffusion state in the first chunk.
    n_steps = gradient.shape[1] - 1

    # Shared waveforms need at most three phase channels. A bounded search for
    # other low-rank waveforms pays off once enough walkers reuse the basis.
    phase_gradient, phase_coefficients = _shared_waveform_basis(
        gradient, allow_bounded=n_walkers >= 4096
    )
    step_l = np.sqrt(6 * diffusivity * dt)
    if phase_coefficients is not None:
        error_bound = _phase_error_upper_bound(
            gradient,
            phase_gradient,
            phase_coefficients,
            dt,
            _position_radius_bounds(substrate, n_steps, step_l, max_iter, epsilon),
            GAMMA,
        )
        if (
            not np.isfinite(error_bound)
            or error_bound > _MAX_PHASE_COMPRESSION_ERROR_RAD
        ):
            phase_gradient, phase_coefficients = gradient, None

    local_to_lab = None
    if substrate.type == "cylinder":
        local_to_lab = utils.vec2vec_rotmat(
            substrate.orientation, np.array([1.0, 0.0, 0.0])
        ).T
    elif substrate.type == "ellipsoid":
        local_to_lab = substrate.R
    if local_to_lab is not None:
        phase_gradient = np.einsum("ktj,ji->kti", phase_gradient, local_to_lab)

    chunk_size = (
        1
        if traj is not None
        else max(
            1,
            int(
                _step_chunk_size(
                    substrate.type, n_walkers, n_steps, phase_gradient.shape[0]
                )
            ),
        )
    )
    if substrate.type not in ("free", "sphere", "cylinder", "ellipsoid", "mesh") or (
        substrate.type != "free" and phase_gradient.shape[0] > 16
    ):
        chunk_size = 1
    diffusion_seed = np.uint64(seed)
    rng_states = (
        empty_xoroshiro128p_states(gs * bs, stream)
        if chunk_size > 1
        else create_local_xoroshiro128p_states(
            gs * bs, seed=diffusion_seed, stream=stream
        )
    )
    d_g_x = cuda.to_device(np.ascontiguousarray(phase_gradient[:, :, 0]), stream=stream)
    d_g_y = cuda.to_device(np.ascontiguousarray(phase_gradient[:, :, 1]), stream=stream)
    d_g_z = cuda.to_device(np.ascontiguousarray(phase_gradient[:, :, 2]), stream=stream)
    d_phases = cuda.to_device(
        np.zeros((phase_gradient.shape[0], n_walkers)), stream=stream
    )
    d_iter_exc = cuda.to_device(np.zeros(n_walkers, dtype=np.bool_), stream=stream)
    trajectory = (
        _TrajectoryRecorder(traj, n_walkers, n_steps, gs, bs, stream, local_to_lab)
        if traj is not None
        else None
    )

    if not quiet:
        print("Number of random walkers = %s" % n_walkers)
        print("Number of steps = %s" % (gradient.shape[1] - 1))
        print("Step length = %s m" % step_l)
        print("Step duration = %s s" % dt)

    # Prepare positions and choose the specialized compiled kernels once.
    initial_host_positions = None
    if substrate.type == "free":
        d_positions = _device_initial_positions("free", n_walkers, seed, bs, stream)
        step_kernel = _cuda_step_free
        fused_kernel = (
            _cuda_fused_free_small_phase
            if phase_gradient.shape[0] <= 16
            else _cuda_fused_free_large_phase
        )
        step_args = ()
        fused_args = ()

    elif substrate.type == "sphere":
        d_positions = _device_initial_positions(
            "sphere", n_walkers, seed, bs, stream, radius=substrate.radius
        )
        step_kernel = _cuda_step_sphere
        fused_kernel = _cuda_fused_sphere_small_phase
        step_args = (substrate.radius, d_iter_exc, max_iter, epsilon)
        fused_args = step_args

    elif substrate.type == "cylinder":
        d_positions = _device_initial_positions(
            "cylinder", n_walkers, seed, bs, stream, radius=substrate.radius
        )
        step_kernel = _cuda_step_cylinder_local
        fused_kernel = _cuda_fused_local_analytic_small_phase
        step_args = (substrate.radius, d_iter_exc, max_iter, epsilon)
        fused_args = (
            1,
            substrate.radius,
            d_positions[0, :],
            d_iter_exc,
            max_iter,
            epsilon,
        )

    elif substrate.type == "ellipsoid":
        d_semiaxes = cuda.to_device(substrate.semiaxes, stream=stream)
        d_positions = _device_initial_positions(
            "ellipsoid", n_walkers, seed, bs, stream, dimensions=d_semiaxes
        )
        step_kernel = _cuda_step_ellipsoid_local
        fused_kernel = _cuda_fused_local_analytic_small_phase
        step_args = (d_semiaxes, d_iter_exc, max_iter, epsilon)
        fused_args = (2, 0.0, d_semiaxes, d_iter_exc, max_iter, epsilon)

    elif substrate.type == "mesh":
        device_mesh = _upload_mesh(
            substrate,
            stream,
            classification=substrate.init_pos in ("intra", "extra")
            if isinstance(substrate.init_pos, str)
            else False,
        )
        if isinstance(substrate.init_pos, np.ndarray):
            if n_walkers != substrate.init_pos.shape[0]:
                raise ValueError(
                    "n_walkers must be equal to the number of initial positions"
                )
            # Mesh kernels work in float64 even for explicit float32 starts.
            initial_host_positions = np.ascontiguousarray(
                substrate.init_pos, dtype=np.float64
            )
            d_positions = cuda.to_device(initial_host_positions, stream=stream)
        else:
            if not quiet:
                print("Calculating initial positions")
            if substrate.init_pos == "uniform":
                d_positions = _device_initial_positions(
                    "uniform_voxel",
                    n_walkers,
                    seed,
                    bs,
                    stream,
                    dimensions=device_mesh["voxel_size"],
                )
            else:
                d_positions = _fill_mesh(
                    n_walkers,
                    substrate,
                    substrate.init_pos == "intra",
                    seed,
                    cuda_bs=bs,
                    stream=stream,
                    device_mesh=device_mesh,
                    return_device=True,
                )
            if not quiet:
                print("Finished calculating initial positions")

        mesh_args = (
            device_mesh["triangle_data"],
            device_mesh["xs"],
            device_mesh["ys"],
            device_mesh["zs"],
            device_mesh["subvoxel_indices"],
            device_mesh["triangle_indices"],
            d_iter_exc,
            max_iter,
            device_mesh["n_sv"],
            epsilon,
            substrate.perm_prob,
            substrate.n_membrane_faces,
        )
        step_kernel = _cuda_step_mesh
        fused_kernel = _cuda_fused_mesh_small_phase
        step_args = mesh_args
        fused_args = mesh_args

    else:
        raise ValueError("Incorrect value (%s) for substrate" % substrate)

    if trajectory is not None:
        trajectory.write_initial(d_positions, initial_host_positions)

    # Launch the selected kernel through one execution loop per RNG mode.
    common_args = (d_positions, d_g_x, d_g_y, d_g_z, d_phases, rng_states)
    kernel = (fused_kernel if chunk_size > 1 else step_kernel)[gs, bs, stream]
    if chunk_size > 1:
        for start in range(0, n_steps, chunk_size):
            stop = min(start + chunk_size, n_steps)
            kernel(
                *common_args,
                start,
                stop,
                diffusion_seed,
                start == 0,
                step_l,
                dt,
                *fused_args,
            )
            if not quiet:
                sys.stdout.write(f"\r{np.round(stop / n_steps * 100, 1)}%")
                sys.stdout.flush()
    else:
        for t in range(n_steps):
            kernel(*common_args, t, step_l, dt, *step_args)
            if trajectory is not None:
                trajectory.record(d_positions)
            if not quiet:
                sys.stdout.write(f"\r{np.round((t + 1) / n_steps * 100, 1)}%")
                sys.stdout.flush()

    if trajectory is not None:
        trajectory.close()

    # A partially resolved collision corrupts both the final position and the
    # phase, so fail the whole run instead of silently dropping walkers.
    if substrate.type != "free":
        iter_exc = d_iter_exc.copy_to_host(stream=stream)
        stream.synchronize()
        failed = np.flatnonzero(iter_exc)
        if failed.size:
            preview = failed[:20].tolist()
            suffix = "..." if failed.size > len(preview) else ""
            raise RuntimeError(
                f"Collision resolution exceeded max_iter={max_iter} for "
                f"{failed.size} walker(s): {preview}{suffix}. Increase max_iter "
                "or reduce the diffusion step length."
            )

    # Aggregate signals on the GPU. Only explicit per-walker output transfers
    # a phase matrix to the host.
    if all_signals:
        phases = d_phases.copy_to_host(stream=stream)
        stream.synchronize()
        if phase_coefficients is not None:
            phases = phase_coefficients @ phases
        signals = np.cos(phases)
    else:
        signals = _reduce_phases(
            d_phases, phase_coefficients, gradient.shape[0], False, stream
        )
    if not quiet:
        sys.stdout.write("\rSimulation finished\n")
        sys.stdout.flush()
    if final_pos:
        positions = d_positions.copy_to_host(stream=stream)
        stream.synchronize()
        if local_to_lab is not None:
            positions = positions @ local_to_lab.T
        return signals, positions
    else:
        return signals
