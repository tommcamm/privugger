import os
import torch
import pyro
import unittest
import pytest
import numpy as np

import privugger as pv
from privugger.inference.pyro.inference import generate_auto_guide, infer_pyro
from privugger.inference.pyro.models import generate_model

# Import example programs using relative paths that work cross-platform
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
program_identity = os.path.join(BASE_DIR, "test", "example_programs", "basic", "identity.py")

@pytest.mark.pyro
class TestPyroAutoGuide(unittest.TestCase):
    """
    Tests for the AutoGuide functionality in Pyro backend.
    """
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_auto_guide_generation(self):
        """
        Test that AutoGuide can be generated from a model.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create model
        model = generate_model(lambda x: x, [age], name="model")
        
        # Generate auto guide
        guide = generate_auto_guide(model, target_names=["age"])
        
        # Check that guide is callable
        self.assertTrue(callable(guide))
        
        # Test that the guide can be called
        guide()
        
        # Check that param store contains parameters
        self.assertTrue(len(pyro.get_param_store()) > 0)
    
    def test_svi_with_auto_guide_basic(self):
        """
        Test SVI inference with AutoGuide on a simple model.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with SVI method and autoguide
        trace = infer_pyro(
            prog, 
            [age], 
            output_type=pv.Float, 
            num_steps=300,
            num_samples=500,
            method="svi",
            autoguide=True,
            chains=2
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
    
    def test_auto_guide_with_observation(self):
        """
        Test AutoGuide with observation constraints.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add observation that output is at least 35
        prog.add_observation("output >= 35", precision=0.1)
        
        # Run inference with SVI method and autoguide
        trace = infer_pyro(
            prog, 
            [age], 
            output_type=pv.Float, 
            num_steps=300,
            num_samples=500,
            method="svi",
            autoguide=True,
            chains=2
        )
        
        # Extract posterior samples for the target variable
        age_posterior = trace.posterior["age"].values
        
        # Check that most samples satisfy the constraint
        self.assertTrue(np.mean(age_posterior >= 35) > 0.8)
        
        # Mean should be shifted upward
        self.assertGreater(np.mean(age_posterior), 35)
    
    def test_hybrid_with_auto_guide(self):
        """
        Test hybrid inference with AutoGuide.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=30.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Run inference with hybrid method and autoguide
        trace = infer_pyro(
            prog, 
            [age], 
            output_type=pv.Float, 
            num_steps=300,
            num_samples=500,
            method="hybrid",
            autoguide=True,
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
    
    def test_complex_model_with_autoguide(self):
        """
        Test AutoGuide on a more complex model with multiple variables.
        """
        # Define multiple priors
        age = pv.Normal("age", mu=30.0, std=5.0)
        height = pv.Normal("height", mu=170.0, std=10.0)
        weight = pv.Normal("weight", mu=70.0, std=15.0)
        
        # Create dataset and program for BMI calculation
        ds = pv.Dataset(input_specs=[age, height, weight])
        
        def bmi_func(age, height, weight):
            return weight / ((height / 100) ** 2)
        
        prog = pv.Program("bmi", dataset=ds, output_type=pv.Float, function=bmi_func)
        
        # Add a constraint that BMI is in healthy range
        prog.add_observation("bmi >= 20 && bmi <= 25", precision=0.5)
        
        # Run inference with SVI method and autoguide
        trace = infer_pyro(
            prog, 
            [age, height, weight], 
            output_type=pv.Float, 
            num_steps=300,
            num_samples=500,
            method="svi",
            autoguide=True,
            chains=2,
            target_idx=2  # Focus on weight
        )
        
        # Extract posterior samples
        weight_posterior = trace.posterior["weight"].values
        height_posterior = trace.posterior["height"].values
        
        # Calculate BMI for all samples
        bmi = weight_posterior / ((height_posterior / 100) ** 2)
        
        # Check that most samples satisfy the BMI constraint
        constraint_satisfaction = np.mean((bmi >= 20) & (bmi <= 25))
        self.assertGreater(constraint_satisfaction, 0.5)
        
        # Print statistics for debugging
        print(f"Constraint satisfaction rate with AutoGuide: {constraint_satisfaction:.2f}")

if __name__ == '__main__':
    unittest.main()
