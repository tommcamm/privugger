import os
import torch
import pyro
import unittest
import pytest
import numpy as np

import privugger as pv
from privugger.inference.pyro.inference import run_mcmc, infer_pyro
from privugger.inference.pyro.models import generate_model

# Import example programs using relative paths that work cross-platform
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
program_identity = os.path.join(BASE_DIR, "test", "example_programs", "basic", "identity.py")

@pytest.mark.pyro
class TestPyroMCMCInference(unittest.TestCase):
    """
    Tests for the MCMC (NUTS) inference in Pyro backend.
    """
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_mcmc_inference_basic(self):
        """
        Test basic MCMC inference with a simple identity function.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with MCMC method
        trace = infer_pyro(
            prog, 
            [age], 
            output_type=pv.Float, 
            num_samples=500,
            method="mcmc",
            chains=2,
            warmup_steps=100
        )
        
        # Check that the posterior has the expected shape
        self.assertIn("age", trace.posterior)
        
        # Extract posterior samples for the target variable
        age_posterior = trace.posterior["age"].values
        
        # Get mean and std of posterior samples
        age_mean = np.mean(age_posterior)
        age_std = np.std(age_posterior)
        
        # For the identity function, posterior should be similar to prior
        self.assertAlmostEqual(age_mean, 30.0, delta=5.0)
        self.assertAlmostEqual(age_std, 5.0, delta=3.0)
    
    def test_mcmc_with_observation(self):
        """
        Test MCMC inference with observation constraints.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add observation that output is at least 35
        prog.add_observation("output >= 35", precision=0.1)
        
        # Run inference with MCMC method
        trace = infer_pyro(
            prog, 
            [age], 
            output_type=pv.Float, 
            num_samples=500,
            method="mcmc",
            chains=2,
            warmup_steps=100
        )
        
        # Extract posterior samples for the target variable
        age_posterior = trace.posterior["age"].values
        
        # Check that most samples satisfy the constraint
        self.assertTrue(np.mean(age_posterior >= 35) > 0.8)
        
        # Mean should be shifted upward
        self.assertGreater(np.mean(age_posterior), 35)
    
    def test_run_mcmc_function(self):
        """
        Test the run_mcmc function directly.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create model
        model = generate_model(lambda x: x, [age], name="model")
        
        # Run MCMC
        _, samples = run_mcmc(
            model, 
            num_samples=500, 
            num_chains=2, 
            warmup_steps=100
        )
        
        # Check that we have samples for the target variable
        self.assertIn("age", samples)
        
        # Convert to numpy for easier analysis
        age_samples = samples["age"].detach().numpy()
        
        # Check shape and basic statistics
        self.assertEqual(age_samples.shape[0], 500)  # 500 samples
        self.assertAlmostEqual(np.mean(age_samples), 30.0, delta=5.0)

if __name__ == '__main__':
    unittest.main()
