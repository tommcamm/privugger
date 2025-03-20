import os
import time
import numpy as np
import torch
import pyro
import unittest
import pytest
import arviz as az

import privugger as pv

# Import example programs using relative paths that work cross-platform
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
program_identity = os.path.join(BASE_DIR, "test", "example_programs", "basic", "identity.py")
program_addition = os.path.join(BASE_DIR, "test", "example_programs", "basic", "addition.py")
program_multiplication = os.path.join(BASE_DIR, "test", "example_programs", "basic", "multiplication.py")

@pytest.mark.pyro
class TestPyroBackendComparison(unittest.TestCase):
    """
    Tests that compare the Pyro backend with other inference backends in Privugger.
    
    These tests focus on ensuring that results from different backends are
    reasonably consistent with each other, which helps validate the correctness
    of the Pyro implementation.
    """
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    @pytest.mark.slow
    def test_backend_comparison(self):
        """
        Compare PyMC and Pyro backends on the same simple model
        
        This test verifies that both backends produce similar posterior
        distributions for the same model with the same priors.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=25.0, std=3.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with both backends
        pymc_trace = pv.infer(prog, draws=1000, method="pymc3")
        
        # Reset for the next inference
        pv.reset()
        
        # Run with Pyro backend
        pyro_trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=500)
        
        # Compare the means of the posteriors
        pymc_mean = pymc_trace.posterior["output"].values.mean()
        pyro_mean = pyro_trace.posterior["output"].values.mean()
        
        # The means should be relatively close
        self.assertAlmostEqual(pymc_mean, pyro_mean, delta=3.0)
        
        # Also check the standard deviations
        pymc_std = pymc_trace.posterior["output"].values.std()
        pyro_std = pyro_trace.posterior["output"].values.std()
        
        # The standard deviations should also be relatively close
        self.assertAlmostEqual(pymc_std, pyro_std, delta=2.0)
    
    @pytest.mark.slow
    def test_performance_comparison(self):
        """
        Compare runtime performance between PyMC and Pyro backends
        
        This test is more informational than a strict assertion.
        It helps track the relative performance of the Pyro backend.
        """
        import time
        
        # Define a more complex model for performance testing
        a = pv.Normal("age", mu=30.0, std=5.0)
        b = pv.Normal("height", mu=170.0, std=10.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[a, b])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_addition)
        
        # Time PyMC inference
        pymc_start = time.time()
        pymc_trace = pv.infer(prog, draws=500, method="pymc3")
        pymc_time = time.time() - pymc_start
        
        # Reset for the next inference
        pv.reset()
        
        # Time Pyro inference
        pyro_start = time.time()
        pyro_trace = pv.infer(prog, draws=500, method="pyro", svi_steps=300)
        pyro_time = time.time() - pyro_start
        
        # Print performance comparison (not a strict test assertion)
        print(f"Performance comparison:")
        print(f"  PyMC time: {pymc_time:.3f} seconds")
        print(f"  Pyro time: {pyro_time:.3f} seconds")
        
        # Verify both methods produced valid results
        self.assertIn("output", pymc_trace.posterior)
        self.assertIn("output", pyro_trace.posterior)


if __name__ == '__main__':
    unittest.main()
