# Development plan for v0.4.0

Status: in progress. Working branch: `dev`. Updated: 7 October 2026.

## Scope

- Target Python 3.11–3.14, CUDA 12–13, and NVIDIA compute capability 7.5 or higher. Record tested combinations; the target does not imply full matrix coverage.
- Preserve public imports, function signatures, SI units, signal shapes, and numerical meaning. Require repeatability for each fixed seed; document any change in exact random sequences.
- Modernize and test the current Numba-CUDA implementation first. Migrate to Numba-CUDA-MLIR as the final implementation step before release validation.
- Complete each item in a small, reviewable commit. Check it off when its stated test or evidence is recorded. Keep GPU-dependent items open until a GPU machine is available.

## Current evidence

- The `disimpy` Conda environment runs Python 3.11. The fast CPU and simulator suite passes: 26 tests, with 23 full-suite tests deselected by `-m 'not gpu'`.
- The wheel and source distribution build and install locally. The documentation builds with warnings treated as errors. CI is defined but has not run on GitHub.
- The full numerical suite and execution on a physical GPU have not been tested.

## Work items

### 1. Baseline and CPU simulation

- [x] Record reference outputs and tolerances for free, sphere, cylinder, ellipsoid, and mesh substrates.
- [x] Add small CUDA simulator tests for shapes, zero and nonzero gradients, confinement, and seeded repeatability. Set `NUMBA_ENABLE_CUDASIM=1` before import; keep walker and step counts small.
- [x] Separate long statistical GPU tests from fast tests and write temporary trajectories outside the repository.

**Gate:** CPU and small simulator tests pass locally on Python 3.11.

### 2. Packaging and CI

- [x] Add `pyproject.toml` with separate runtime, test, and documentation dependencies; build and install wheel and source distribution.
- [x] Use one version source for package and documentation; replace the exported Conda environment with a portable specification.
- [ ] Verify CI on Python 3.11–3.14 for CPU and simulator tests, installation, formatting, linting, and documentation. Apply style checks to new and refactored modules. Keep `disimpy.tests.test_all()` available through the test extra.

**Gate:** clean CI runs pass on all four Python versions.

### 3. Modularization

- [ ] Move CPU initialization and mesh preparation into private modules without changing results.
- [ ] Separate CUDA device functions and kernels from host execution, transfers, and output.
- [x] Load plotting dependencies only when plotting; retain public utility functions.
- [ ] Scope warning handling to the operations that need it; retain public imports and `simulations.simulation()` arguments.

**Gate:** each change passes the fast tests and matches the recorded baseline.

### 4. Optimization on the current backend

- [ ] Benchmark and optimize CPU preparation, including mesh subdivision and gradient construction.
- [ ] Inspect transfers, synchronization, and trajectory storage; change only measured bottlenecks.
- [ ] Benchmark Numba-CUDA execution on the GPU laptop when available. Do not use the simulator for performance claims.

**Gate:** each accepted change has a before-and-after measurement and a numerical regression result.

### 5. MLIR migration

- [ ] Pin a released Numba-CUDA-MLIR version and update the release dependency.
- [ ] On the GPU laptop, test kernels, local arrays, device helpers, streams, transfers, and random-state initialization before porting each substrate.
- [ ] Verify fixed-seed repeatability and numerical tolerances. Isolate the MLIR compatibility random API, which is deprecated.
- [ ] Add GPU-free MLIR compilation checks where feasible. Retain Numba-CUDA simulator coverage after migration only if a small test adapter can run the same kernel source.

**Gate:** every substrate executes correctly with MLIR on a physical GPU. Resolve blockers or review the release scope explicitly.

### 6. Validation and release

- [ ] Run the full numerical suite and repeat representative benchmarks with MLIR on the GPU laptop. Record GPU, driver, Python, CUDA, and MLIR versions.
- [ ] Document untested target combinations, installation, compatibility, and numerical or performance changes.
- [ ] Prepare `0.4.0rc1`; inspect installed wheel and source distribution. Publish v0.4.0 from a reviewed tag through GitHub and PyPI trusted publishing.

**Release gate:** CPU CI passes; MLIR numerical tests pass on at least one GPU with compute capability 7.5 or higher; public examples, documentation, artifacts, and versions agree. Every performance claim has a benchmark and regression result.
