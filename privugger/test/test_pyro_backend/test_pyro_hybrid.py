import os
import torch
import pyro
import unittest
import pytest
import numpy as np

import privugger as pv
from privugger.inference.pyro.inference import run_hybrid, infer_pyro
from privugger.inference.pyro.models import generate_model, generate_guide

# Import example programs using relative paths that work cross-platform
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
program_identity = os.path.join(BASE_DIR, "test", "example_programs", "basic", "identity.py")

@pytest.mark.pyro
class TestPyroHybridInference(unittest.TestCase):
    """
    Tests for the hybrid SVI+MCMC inference in Pyro backend.
    """
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_hybrid_inference_basic(self):
        """
        Test hybrid inference with a simple identity function.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with hybrid method
        trace = infer_pyro(
            prog, 
            [age], 
            output_type=pv.Float, 
            num_steps=300,  # SVI steps
            num_samples=500,  # MCMC samples
            method="hybrid",
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
    
    def test_hybrid_with_observation(self):
        """
        Test hybrid inference with observation constraints.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add observation that output is at least 35
        prog.add_observation("output >= 35", precision=0.1)
        
        # Run inference with hybrid method
        trace = infer_pyro(
            prog, 
            [age], 
            output_type=pv.Float, 
            num_steps=300,  # SVI steps
            num_samples=500,  # MCMC samples
            method="hybrid",
            chains=2,
            warmup_steps=100
        )
        
        # Extract posterior samples for the target variable
        age_posterior = trace.posterior["age"].values
        
        # Check that most samples satisfy the constraint
        self.assertTrue(np.mean(age_posterior >= 35) > 0.8)
        
        # Mean should be shifted upward
        self.assertGreater(np.mean(age_posterior), 35)
    
    def test_run_hybrid_function(self):
        """
        Test the run_hybrid function directly.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create model and guide
        model = generate_model(lambda x: x, [age], name="model")
        guide = generate_guide([age], target_idx=0)
        
        # Run hybrid inference
        _, samples = run_hybrid(
            model, 
            guide,
            num_svi_steps=300,
            lr=0.01,
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
    
    def test_hybrid_vs_mcmc_efficiency(self):
        """
        Test that hybrid inference is more efficient than pure MCMC.
        
        This test measures the quality of posterior approximation
        for a fixed computational budget.
        """
        # Define a simple but slightly more complex model
        age = pv.Normal("age", mu=30.0, std=5.0)
        height = pv.Normal("height", mu=170.0, std=10.0)
        
        # Create dataset and program for BMI calculation
        ds = pv.Dataset(input_specs=[age, height])
        
        # Simple BMI-like calculation (not actual BMI but similar concept)
        def bmi_func(age, height):
            # Using age as a weight proxy for this test
            return (age * 3) / ((height / 100) ** 2)
        
        prog = pv.Program("bmi", dataset=ds, output_type=pv.Float, function=bmi_func)
        
        # Add a constraint that BMI is in healthy range
        prog.add_observation("bmi >= 20 && bmi <= 25", precision=0.5)
        
        # Run inference with both methods
        hybrid_trace = infer_pyro(
            prog, 
            [age, height], 
            output_type=pv.Float, 
            num_steps=200,  # SVI steps
            num_samples=300,  # MCMC samples
            method="hybrid",
            chains=2,
            warmup_steps=50,
            target_idx=0  # Focus on age variable
        )
        
        mcmc_trace = infer_pyro(
            prog, 
            [age, height], 
            output_type=pv.Float, 
            num_samples=300,  # MCMC samples
            method="mcmc",
            chains=2,
            warmup_steps=100,  # More warmup steps for pure MCMC
            target_idx=0  # Focus on age variable
        )
        
        # Extract posterior samples for the target variable
        hybrid_age = hybrid_trace.posterior["age"].values
        mcmc_age = mcmc_trace.posterior["age"].values
        
        # Calculate BMI for all samples to check constraint satisfaction
        hybrid_height = hybrid_trace.posterior["height"].values
        mcmc_height = mcmc_trace.posterior["height"].values
        
        hybrid_bmi = (hybrid_age * 3) / ((hybrid_height / 100) ** 2)
        mcmc_bmi = (mcmc_age * 3) / ((mcmc_height / 100) ** 2)
        
        # Calculate constraint satisfaction rates
        hybrid_satisfaction = np.mean((hybrid_bmi >= 20) & (hybrid_bmi <= 25))
        mcmc_satisfaction = np.mean((mcmc_bmi >= 20) & (mcmc_bmi <= 25))
        
        # Hybrid method should have comparable or better constraint satisfaction
        self.assertGreaterEqual(hybrid_satisfaction * 0.8, mcmc_satisfaction * 0.8)
        
        # Both methods should have reasonable satisfaction rates
        self.assertGreater(hybrid_satisfaction, 0.5)
        self.assertGreater(mcmc_satisfaction, 0.5)
        
        # Print statistics for debugging
        print(f"Hybrid constraint satisfaction: {hybrid_satisfaction:.2f}")
        print(f"MCMC constraint satisfaction: {mcmc_satisfaction:.2f}")

if __name__ == '__main__':
    unittest.main()
