************
Contributing
************

Contributions to Disimpy are warmly welcomed.

If you want to discuss ideas before writing code, please open an `issue on
GitHub <https://github.com/kerkelae/disimpy/issues>`_, and we'll discuss how to
continue. GitHub issues are also a great way to inform the developers about
any bugs or problems you may encounter.

Development workflow
####################

If you already have a well-defined idea or want to make small changes (fix
typos, improve documentation, etc.), please follow the steps below:

1. Fork the `repository on GitHub <https://github.com/kerkelae/disimpy/>`_.
2. Clone your fork:

.. code-block::

    git clone git@github.com:YOUR-USERNAME/disimpy.git

3. Configure Git to sync your fork with the main repo:

.. code-block::

    git remote add upstream https://github.com/kerkelae/disimpy.git

4. Create a branch with a name that describes your contribution:

.. code-block::

    git checkout -b BRANCH-NAME

5. Write code, commit changes, and push to your fork on GitHub.

6. Open a pull request `on GitHub <https://github.com/kerkelae/disimpy/>`_.

GitHub docs provide more information on `forking a repository
<https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ and `creating
pull requests
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/
proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-
a-fork>`_.

Development environment
#######################

Use a virtual environment for development. The provided Conda environment
selects Python 3.11; other supported Python versions also work:

.. code-block:: console

   conda env create -f development_environment.yml
   conda activate disimpy
   python -m pip install -r requirements-dev.txt

Run these commands from the repository root. ``requirements-dev.txt`` installs
Disimpy in editable mode with its test and documentation extras, plus Ruff,
build, and Twine. These tools are separate from the package's runtime
requirements. The fast CPU and CUDA simulator tests do not need a CUDA Toolkit
or NVIDIA driver.

For physical GPU simulations, install the CUDA Toolkit components for your
target version as well:

.. code-block:: console

   python -m pip install -e '.[cuda13]'

Use ``cuda12`` in place of ``cuda13`` when targeting CUDA 12.

Code style
##########

Format new and refactored Python modules with Ruff. Use the `NumPy docstring
conventions <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for
public functions.

Tests
#####

Run the fast CPU and CUDA simulator tests without a GPU from the repository
root:

.. code-block::

   NUMBA_ENABLE_CUDASIM=1 python -m pytest -q -m 'not gpu'

The GitHub Actions test job runs this lightweight CPU and CUDA simulator
suite. Before merging code, the developer is responsible for verifying
that Disimpy still runs on a supported physical GPU. Install the appropriate
``cuda12`` or ``cuda13`` extra, disable the simulator, and run the full
numerical suite:

.. code-block::

   env -u NUMBA_ENABLE_CUDASIM python -m pytest -q

Documentation
#############

After installing the documentation extra, build the documentation from the
repository root. Install Pandoc first (for example, ``conda install
-c conda-forge pandoc``) so nbsphinx can convert the tutorial notebooks:

.. code-block::

    sphinx-build -b html -W --keep-going docs/source docs/_build/html

The generated HTML is in ``docs/_build/html``.

Releasing
#########

Follow the `release guide
<https://github.com/kerkelae/disimpy/blob/main/RELEASING.md>`_ for version and
tag rules, developer GPU validation, the exact-commit ``main`` CI gate, and
PyPI publishing. Review the release notes and artifacts before publishing the
GitHub Release.
