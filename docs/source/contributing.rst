************
Contributing
************

Contributions to Disimpy are warmly welcomed.

If you want to discuss ideas before writing code, please open an `issue on
GitHub <https://github.com/kerkelae/disimpy/issues>`_ and we'll discuss how to
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

Code style
##########

All code should be formatted with `Black <https://github.com/psf/black>`_ using
the default settings and documented following the `Numpy docstring conventions
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ (except trivial
internal functions).

Tests
#####

It is important to make sure that your changes have not broken
anything by running all tests:

.. code-block:: python

   import disimpy.tests
   disimpy.tests.test_all()

Documentation
#############

If you make changes to the documentation, you should build it locally to
confirm that it works as expected by executing the following in
``disimpy/docs``:

.. code-block::

    make clean
    make html

This will generate a local copy of the documentation in
``disimpy/docs/_build/html``. Once the changes are merged with the master
branch, the online documentation is automatically updated. Generating the
documentation locally requires `sphinx <https://www.sphinx-doc.org/>`_ and
`nbsphinx <https://nbsphinx.readthedocs.io/>`_ that can be installed with pip: 

.. code-block::

    pip install sphinx nbpshinx
