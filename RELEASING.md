# Releasing Disimpy

`disimpy/_version.py` is the version source for the package. `pyproject.toml`
reads that value when building a distribution. Use a matching `v`-prefixed Git
tag: version `0.4.0rc2` has tag `v0.4.0rc2`, and version `0.4.0` has tag
`v0.4.0`. Choose a version that has not already been published on PyPI. The
release workflow rejects development versions containing `.dev`.

## Prepare the release

1. Set `disimpy/_version.py` to the intended release version in a pull
   request, and review the release notes. Merge the change into `main`.
2. Wait for the **CI** workflow's **push** run on the exact `main` commit to
   succeed. A successful pull request run or a run for another commit does not
   satisfy the release gate.
3. Run **Physical GPU validation** from GitHub Actions on `main` after CI
   succeeds. Choose `cuda12` or `cuda13` for the runner's driver. The workflow
   requires a Linux x64 self-hosted runner labeled `disimpy-gpu`, with an
   NVIDIA GPU of compute capability 7.5 or higher, a compatible driver, and
   Python 3.11 available through `actions/setup-python`. It installs the test
   and selected CUDA dependencies, confirms the MLIR backend sees a physical
   GPU, and runs the GPU and shared tests. Simulator-only tests are covered by
   CI. Wait for a successful run on the exact commit to be released.
   GitHub-hosted standard runners do not provide a CUDA GPU.

   To reproduce that check locally on the same commit, use a supported GPU
   machine and run:

   ```sh
   python -m pip install -r requirements-dev.txt
   python -m pip install -e '.[cuda13]'  # use cuda12 for a CUDA 12 environment
   env -u NUMBA_ENABLE_CUDASIM python -m pytest -q
   ```

   CI runs the fast simulator suite on Python 3.11–3.14. Review the physical
   GPU workflow result before publishing.

## Publish

1. Create a Git tag on the tested `main` commit with the matching version,
   then push it. Use the exact commit SHA shown by the successful CI and GPU
   validation runs:

   ```sh
   git fetch origin main
   read -r -p "Exact tested main commit SHA: " tested_sha
   git tag v0.4.0rc2 "$tested_sha"
   git push origin v0.4.0rc2
   ```

2. Create and publish a GitHub Release from that tag with release notes.
   Mark `rc` releases as **pre-releases** in GitHub; leave that option unchecked
   for a stable release. The workflow publishes both kinds when their tag and
   package version match.
3. Check that the **Publish release** workflow succeeds, that PyPI has the
   wheel and source distribution, and that both files are attached to the
   GitHub Release.

The release workflow checks that the tag's commit is on `main`, that the
exact commit has successful CI and physical GPU validation runs on `main`, and
that the tag matches `disimpy/_version.py`. It then repeats the fast simulator
tests, builds and checks the distributions with Twine, publishes to PyPI
through the `pypi` GitHub environment and PyPI Trusted Publishing, and uploads
the same files as GitHub Release assets. Check PyPI's trusted publisher
configuration for the `kerkelae/disimpy` repository,
`.github/workflows/release.yml` workflow, and `pypi` environment before
publishing through this workflow.

Normal installations use the runtime dependencies declared in
`pyproject.toml`. `requirements-dev.txt` installs the test and documentation
extras plus development tools; it is for release preparation and development,
not a package runtime requirement. The `cuda12` and `cuda13` extras select
additional Toolkit components for the chosen GPU environment.
