import os
from privugger.inference.pyro.inference import run_svi
from privugger.inference.pyro.models import generate_guide, generate_model
import torch
import pyro
import unittest
import pytest

import privugger as pv

# Import example programs using relative paths that work cross-platform
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
program_identity = os.path.join(BASE_DIR, "test", "example_programs", "basic", "identity.py")

@pytest.mark.pyro
class TestPyroGuideParameterLearning(unittest.TestCase):
    """
    Tests for the guide generation and parameter learning in Pyro backend.
    
    These tests specifically focus on the variational inference aspects
    of the Pyro backend, checking that guide parameters are properly
    initialized and optimized.
    """
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_guide_parameter_learning(self):
        """
        Test that the guide parameters are being learned properly
        
        This test checks whether the variational parameters are:
        1. Correctly initialized based on the prior
        2. Updated during optimization
        3. Converge to reasonable values after optimization
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
    
    def test_loss_convergence(self):
        """
        Test that the SVI loss converges during optimization
        
        This checks that the optimization process is working correctly
        by verifying that loss decreases during training.
        """
        # Define a simple prior
        age = pv.Normal("age", mu=25.0, std=3.0)
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Create model and guide functions
        model = generate_model(prog.program, [age], name="model")
        guide = generate_guide([age], target_idx=0, name="guide")
        
        # Run SVI with more steps to better observe convergence
        _, losses = run_svi(model, guide, num_steps=500, lr=0.01)
        
        # Check that losses decrease
        # Compare average of first 50 losses with average of last 50 losses
        early_loss_avg = sum(losses[:50]) / 50
        late_loss_avg = sum(losses[-50:]) / 50
        
        # The later losses should be smaller than the early losses
        self.assertLess(late_loss_avg, early_loss_avg)


if __name__ == '__main__':
    unittest.main()
