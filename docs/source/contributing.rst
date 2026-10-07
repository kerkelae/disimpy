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
   editable mode with test and documentation dependencies from the repository
   root:

   .. code-block::

        pip install -e ".[test,docs]"

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
repository root:

.. code-block::

    sphinx-build -b html -W --keep-going docs/source docs/_build/html

The generated HTML is in ``docs/_build/html``.
