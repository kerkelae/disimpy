************
Contributing
************

We warmly welcome contributions to Disimpy from other scientists and
developers.

If you want to discuss ideas before writing code, please open an `issue on
GitHub <https://github.com/kerkelae/disimpy/issues>`_ and we'll help you get
started. Creating issues is also a great way to inform us about any bugs or
problems you may encounter.

If you already have a well-defined idea or want to make small changes (fix
typos, improve code style, etc.), please fork the repository, create a branch
with a descriptive name, make the necessary changes, and create a pull request.

All code should be formatted with `Black <https://github.com/psf/black>`_ using
the default settings and documented following the `Numpy docstring conventions
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ (except trivial
internal functions). The functions should also have unit tests in the
corresponding test module. It is important to make sure that your changes have
not broken anything by running all tests with:

.. code-block:: python

   import disimpy.tests
   disimpy.tests.test_all()

If you make changes to the documentation, you should build it locally to
confirm that it works as expected by executing the following in
``disimpy/docs``:

.. code-block::

    make clean
    make html

This will generate a local copy of the documentation in
``disimpy/docs/_build/html``. Once the changes are merged with the master
branch, the online documentation is automatically updated.
