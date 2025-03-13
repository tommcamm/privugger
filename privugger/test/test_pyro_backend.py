import os
import sys
import inspect
import numpy as np
import torch
import pyro
import unittest
import arviz as az

# Add the parent directory to the path to import privugger
sys.path.append(os.path.join("../.."))

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import privugger as pv
from privugger.inference.pyro_backend import dist_to_pyro, generate_model, generate_guide, run_svi

# Import test programs
program_identity = "privugger/test/identity.py"
program_addition = "privugger/test/addition.py"
program_multiplication = "privugger/test/multiplication.py"

class TestPyroBackend(unittest.TestCase):
    
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
        self.assertEqual(trace.posterior["age"].shape[1], 1000)  # 1000 samples
        self.assertEqual(trace.posterior["output"].shape[1], 1000)  # 1000 samples
        
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
    
    def test_guide_parameter_learning(self):
        """
        Test that the guide parameters are being learned properly
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Create model and guide functions
        model = generate_model(prog.program, [age], name="model")
        guide = generate_guide([age], target_idx=0, name="guide")
        
        # Initial parameter check - guide parameters should not exist yet
        self.assertFalse("mu_age" in pyro.get_param_store())
        self.assertFalse("sigma_age" in pyro.get_param_store())
        
        # Run SVI
        svi, losses = run_svi(model, guide, num_steps=300, lr=0.01)
        
        # Now the parameters should exist
        self.assertTrue("mu_age" in pyro.get_param_store())
        self.assertTrue("sigma_age" in pyro.get_param_store())
        
        # Check that the learned parameters are close to the prior
        learned_mu = pyro.param("mu_age").item()
        learned_sigma = pyro.param("sigma_age").item()
        
        # Since we're using the identity function, the learned parameters
        # should be close to the original distribution parameters
        self.assertAlmostEqual(learned_mu, 30.0, delta=5.0)
        self.assertAlmostEqual(learned_sigma, 5.0, delta=3.0)
    
    def test_backend_comparison(self):
        """
        Compare PyMC and Pyro backends on the same simple model
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

if __name__ == '__main__':
    unittest.main()
