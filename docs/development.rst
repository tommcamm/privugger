Development
===========

This section provides information for contributors and developers working with Privugger.

Testing
-------

Privugger uses pytest for testing. The test suite has been organized into logical groups with test markers to make it easy to run specific subsets of tests.

Running Tests
~~~~~~~~~~~~

We provide a convenient script ``run_tests.py`` that works across all platforms (Windows, Linux, macOS) for running tests:

.. code-block:: bash

   # Run all tests
   python run_tests.py --all

   # Run only Pyro backend tests
   python run_tests.py --pyro

   # Skip slow tests
   python run_tests.py --all --fast

   # Run a specific test file
   python run_tests.py --specific privugger/test/test_pyro_backend/test_pyro_basic.py

   # Run a specific test class or method
   python run_tests.py --specific privugger/test/test_pyro_backend/test_pyro_basic.py::TestPyroBasicInference::test_pyro_uniform

Test Markers
~~~~~~~~~~~~

We use the following pytest markers to organize tests:

- ``pyro``: Tests that use the Pyro backend
- ``slow``: Tests that take a long time to run (e.g., with many inference steps)

You can run tests with specific markers using the pytest command directly:

.. code-block:: bash

   # Run only tests marked with 'pyro'
   pytest -m pyro

   # Run tests marked with 'pyro' but exclude 'slow' tests
   pytest -m "pyro and not slow"

Test Structure
~~~~~~~~~~~~~

The tests are organized as follows:

- ``privugger/test/test_privugger.py``: General tests for the Privugger framework
- ``privugger/test/test_pyro_backend/``: Tests specific to the Pyro backend
  - ``test_pyro_basic.py``: Basic inference tests
  - ``test_pyro_guide.py``: Tests for guide parameter learning
  - ``test_pyro_observations.py``: Tests for observation handling
  - ``test_pyro_comparison.py``: Tests comparing Pyro with other backends

Writing New Tests
~~~~~~~~~~~~~~~~

When writing new tests:

1. Place them in the appropriate directory based on functionality
2. Use the ``@pytest.mark.pyro`` decorator for Pyro-related tests
3. Use the ``@pytest.mark.slow`` decorator for tests that take a long time to run
4. Ensure tests use cross-platform paths with ``os.path.join``

Building Documentation
---------------------

1. Make sure you have pandoc installed: https://pandoc.org/installing.html

2. Install documentation dependencies:

   .. code-block:: bash
   
      pip install sphinx nbsphinx sphinx_rtd_theme

3. If using a conda environment, register a Jupyter kernel for it:

   .. code-block:: bash
   
      # Ensure your conda environment is activated
      conda activate privugger
      
      # Install ipykernel if needed
      conda install ipykernel
      
      # Register the kernel with the same name as your environment
      python -m ipykernel install --user --name privugger --display-name "Python (privugger)"

4. Navigate to the docs directory:

   .. code-block:: bash
   
      cd docs/

5. Build the HTML documentation:

   .. code-block:: bash
   
      make html
