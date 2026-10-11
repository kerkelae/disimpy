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

.. tip::

   When making changes to the code, it is recommended to use the `conda
   environment
   <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_
   in the file ``development_environment.yml`` and to install the package in
   editable mode with test, documentation, and CUDA Toolkit dependencies from
   the repository root:

   .. code-block::

        pip install -e ".[test,docs,cuda13]"

   Use ``cuda12`` for CUDA 12. The fast simulator tests need only
   ``.[test,docs]``.

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

The full numerical suite needs a physical GPU:

.. code-block::

   python -m pytest -q

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

Before publishing, run the full numerical suite on a physical GPU, review
the Python matrix in GitHub Actions, and inspect the installed wheel and
source distribution. The version in ``disimpy/_version.py`` must match the
reviewed Git tag with a ``v`` prefix, for example ``v0.4.0rc1``.

The ``release.yml`` workflow runs the fast suite on Python 3.11–3.14, builds
the distributions, and checks their metadata. Publishing a GitHub release
from that tag then publishes the same distributions to PyPI through a
`trusted publisher <https://docs.pypi.org/trusted-publishers/using-a-publisher/>`_
and attaches them to the GitHub release. Configure the PyPI trusted publisher
for the ``kerkelae/disimpy`` repository, ``release.yml`` workflow, and
``pypi`` environment before publishing the first release. Review the tag and
artifacts before publishing the GitHub release.
