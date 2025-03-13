Development
===========

This section provides information for contributors and developers working with Privugger.

Testing
-------

To run the unit tests for Privugger, use the following command:

.. code-block:: bash

   python -m privugger.test.test_pyro_backend

To run a specific test case:

.. code-block:: bash

   python -m privugger.test.test_pyro_backend TestPyroBackend.test_pyro_uniform

Building Documentation
---------------------

1. Make sure you have pandoc installed: https://pandoc.org/installing.html

2. Install documentation dependencies:

   .. code-block:: bash
   
      pip install sphinx nbsphinx sphinx_rtd_theme

3. Navigate to the docs directory:

   .. code-block:: bash
   
      cd docs/

4. Build the HTML documentation:

   .. code-block:: bash
   
      make html
