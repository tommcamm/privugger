import os
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
class TestPyroBasicInference(unittest.TestCase):
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_basic_svi(self):
        """
        Test a basic SVI run with the Pyro backend
        """
        # Define a simple prior distribution
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with Pyro backend
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=500)
        
        # Check that we have the expected outputs
        self.assertIn("age", trace.posterior)
        self.assertIn("output", trace.posterior)
        
        # Check shapes are correct
        self.assertEqual(trace.posterior["age"].shape[1], 500)  # 500 samples per chain
        self.assertEqual(trace.posterior["output"].shape[1], 500)  # 500 samples per chain
        
        # Check that output is approximately equal to age (since we used the identity function)
        np.testing.assert_allclose(
            trace.posterior["age"].values.mean(), 
            trace.posterior["output"].values.mean(), 
            rtol=0.1
        )
    
    def test_pyro_normal_addition(self):
        """
        Test the Pyro backend with a simple addition program
        """
        # Define priors
        a = pv.Normal("age", mu=10.0, std=2.0)
        b = pv.Normal("height", mu=40.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[a, b])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_addition)
        
        # Run inference with Pyro backend
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=500)
        
        # Check that the mean of the output is approximately equal to the sum of means
        np.testing.assert_allclose(
            trace.posterior["output"].values.mean(), 
            trace.posterior["age"].values.mean() + trace.posterior["height"].values.mean(),
            rtol=0.1
        )
    
    def test_pyro_uniform(self):
        """
        Test the Pyro backend with a Uniform distribution
        """
        # Define a simple uniform prior
        x = pv.Uniform("x", lower=0.0, upper=10.0)
        
        # Create dataset and program with the identity function
        ds = pv.Dataset(input_specs=[x])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with Pyro backend
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=500)
        
        # Check that we have the expected outputs
        self.assertIn("x", trace.posterior)
        self.assertIn("output", trace.posterior)
        
        # Check shapes
        self.assertEqual(trace.posterior["x"].shape[1], 500)  # 500 samples per chain
        
        # Check that samples are within the bounds
        x_samples = trace.posterior["x"].values.flatten()
        self.assertGreaterEqual(np.min(x_samples), 0.0)
        self.assertLessEqual(np.max(x_samples), 10.0)
        
        # Check that output is approximately equal to x (since we used the identity function)
        np.testing.assert_allclose(
            trace.posterior["x"].values.mean(), 
            trace.posterior["output"].values.mean(), 
            rtol=0.1
        )


if __name__ == '__main__':
    unittest.main()
