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

@pytest.mark.pyro
class TestPyroAutoLR(unittest.TestCase):
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_auto_lr(self):
        """
        Test that auto learning rate works properly with SVI
        """
        # Define a simple prior distribution
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with Pyro backend using auto learning rate
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=500, svi_lr="auto")
        
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
    
    def test_auto_lr_with_observations(self):
        """
        Test that auto learning rate works with observations
        """
        # Define a simple prior distribution
        a = pv.Normal("age", mu=10.0, std=2.0)
        b = pv.Normal("height", mu=40.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[a, b])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_addition)
        
        # Add an observation - this makes inference more challenging
        prog.observation = "output == 55.0"
        prog.observation_precision = 0.01
        
        # Run inference with auto learning rate
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=800, svi_lr="auto")
        
        # Check that we have the expected outputs
        self.assertIn("age", trace.posterior)
        self.assertIn("height", trace.posterior)
        self.assertIn("output", trace.posterior)
        
        # Check that output is close to the observed value
        self.assertAlmostEqual(trace.posterior["output"].values.mean(), 55.0, delta=1.0)
    
    def test_hybrid_with_auto_lr(self):
        """
        Test that auto learning rate works with hybrid inference method
        """
        # Define a simple prior distribution
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run hybrid inference with auto learning rate
        trace = pv.infer(prog, draws=500, method="pyro", 
                         pyro_method="hybrid", svi_steps=300, 
                         svi_lr="auto", warmup_steps=100)
        
        # Check that we have the expected outputs
        self.assertIn("age", trace.posterior)
        self.assertIn("output", trace.posterior)
        
        # Check that output is approximately equal to age (since we used the identity function)
        np.testing.assert_allclose(
            trace.posterior["age"].values.mean(), 
            trace.posterior["output"].values.mean(), 
            rtol=0.1
        )

if __name__ == '__main__':
    unittest.main()
