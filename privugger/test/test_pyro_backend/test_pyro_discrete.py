import os
import torch
import pyro
import unittest
import pytest
import numpy as np

import privugger as pv
from privugger.inference.pyro.inference import run_svi, get_posterior_samples, pyro_to_arviz
from privugger.inference.pyro.models import generate_guide, generate_model

# Import example programs using relative paths that work cross-platform
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Identity function used for simple tests
def identity_fn(x):
    return x

# Categorical shift function to test categorical distribution handling
def categorical_shift(x):
    # Simple function that shifts categorical values
    # If x is 0, return 0, otherwise return x-1
    # This creates a simple but testable relationship
    if torch.is_tensor(x):
        result = torch.where(x > 0, x - 1, x)
        return result
    else:
        return 0 if x == 0 else x - 1

@pytest.mark.pyro
class TestPyroDiscrete(unittest.TestCase):
    """
    Tests for discrete distribution support in the Pyro backend.
    
    These tests focus on validating that SVI works correctly with
    discrete distributions like Categorical, Bernoulli, etc.
    """
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_categorical_distribution(self):
        """
        Test SVI with Categorical distribution
        """
        # Define a categorical prior with uneven probabilities
        categories = pv.Categorical("categories", p=[0.6, 0.3, 0.1])
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[categories])
        prog = pv.Program("output", dataset=ds, output_type=pv.Int, function=identity_fn)
        
        # Create model and guide functions
        model = generate_model(prog.program, [categories], name="model")
        guide = generate_guide([categories], target_idx=0, name="guide")
        
        # Run SVI
        svi, losses = run_svi(model, guide, num_steps=500, lr=0.01)
        
        # Check convergence
        early_loss_avg = sum(losses[:50]) / 50
        late_loss_avg = sum(losses[-50:]) / 50
        self.assertLess(late_loss_avg, early_loss_avg)
        
        # Get samples from posterior
        num_samples = 2000
        samples = get_posterior_samples(model, guide, num_samples=num_samples)
        
        # Extract categorical samples
        cat_samples = samples["categories"].cpu().numpy()
        
        # Count occurrences of each category
        counts = np.zeros(3)
        for i in range(3):
            counts[i] = np.sum(cat_samples == i)
        
        # Convert to empirical probabilities
        empirical_probs = counts / num_samples
        
        # Check that empirical probabilities are close to the prior probabilities
        # Use a loose tolerance since we have a small number of samples
        np.testing.assert_allclose(empirical_probs, [0.6, 0.3, 0.1], atol=0.1)
    
    def test_bernoulli_distribution(self):
        """
        Test SVI with Bernoulli distribution
        """
        # Define a Bernoulli prior
        binary = pv.Bernoulli("binary", p=0.7)
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[binary])
        prog = pv.Program("output", dataset=ds, output_type=pv.Int, function=identity_fn)
        
        # Create model and guide functions
        model = generate_model(prog.program, [binary], name="model")
        guide = generate_guide([binary], target_idx=0, name="guide")
        
        # Run SVI
        svi, losses = run_svi(model, guide, num_steps=500, lr=0.01)
        
        # Check convergence
        early_loss_avg = sum(losses[:50]) / 50
        late_loss_avg = sum(losses[-50:]) / 50
        self.assertLess(late_loss_avg, early_loss_avg)
        
        # Get samples from posterior
        num_samples = 2000
        samples = get_posterior_samples(model, guide, num_samples=num_samples)
        
        # Extract Bernoulli samples
        bin_samples = samples["binary"].cpu().numpy()
        
        # Compute empirical probability
        empirical_prob = np.mean(bin_samples)
        
        # Check that empirical probability is close to the prior probability
        self.assertAlmostEqual(empirical_prob, 0.7, delta=0.1)
    
    def test_categorical_with_observation(self):
        """
        Test SVI with Categorical distribution and observation constraint
        """
        # Define a categorical prior with uniform probabilities
        categories = pv.Categorical("categories", p=[0.25, 0.25, 0.25, 0.25])
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[categories])
        prog = pv.Program("output", dataset=ds, output_type=pv.Int, function=categorical_shift)
        
        # Add observation: output == 2
        # This should constrain the posterior to favor category 3
        prog.add_observation('output==2', precision=0.01)
        
        # Create model and guide functions
        # Pass the Program object through prog_obj for observations to work correctly
        model = generate_model(prog.program, [categories], name="model", prog_obj=prog)
        guide = generate_guide([categories], target_idx=0, name="guide")
        
        # Run SVI with more steps due to the observation constraint
        svi, losses = run_svi(model, guide, num_steps=1000, lr=0.005)
        
        # Get samples from posterior
        num_samples = 2000
        samples = get_posterior_samples(model, guide, num_samples=num_samples)
        
        # Extract categorical samples
        cat_samples = samples["categories"].cpu().numpy()
        
        # Count occurrences of each category
        counts = np.zeros(4)
        for i in range(4):
            counts[i] = np.sum(cat_samples == i)
        
        # Convert to empirical probabilities
        empirical_probs = counts / num_samples
        
        # Since output==2 observation, and categorical_shift(x) returns x-1 for x>0,
        # category 3 should have the highest probability
        self.assertEqual(np.argmax(empirical_probs), 3)
        
        # Category 3 should have a significantly higher probability than others
        self.assertGreater(empirical_probs[3], 0.5)
    
    def test_multi_element_categorical(self):
        """
        Test SVI with multi-element Categorical distribution
        """
        # Define a categorical prior with multiple elements
        num_elements = 5
        ratings = pv.Categorical("ratings", p=[0.2, 0.2, 0.2, 0.2, 0.2], num_elements=num_elements)
        
        # Create model and guide
        ds = pv.Dataset(input_specs=[ratings])
        prog = pv.Program("output", dataset=ds, output_type=pv.List(pv.Int), function=identity_fn)
        
        # Create model and guide functions
        model = generate_model(prog.program, [ratings], name="model")
        guide = generate_guide([ratings], target_idx=0, name="guide")
        
        # Run SVI
        svi, losses = run_svi(model, guide, num_steps=500, lr=0.01)
        
        # Check convergence
        early_loss_avg = sum(losses[:50]) / 50
        late_loss_avg = sum(losses[-50:]) / 50
        self.assertLess(late_loss_avg, early_loss_avg)
        
        # Get samples from posterior
        num_samples = 1000
        samples = get_posterior_samples(model, guide, num_samples=num_samples)
        
        # Extract categorical samples
        rating_samples = samples["ratings"].cpu().numpy()
        
        # Check shape of samples
        self.assertEqual(rating_samples.shape, (num_samples, num_elements))
        
        # Check values are valid categories (0-4)
        self.assertTrue(np.all(rating_samples >= 0))
        self.assertTrue(np.all(rating_samples <= 4))
        
        # Convert to ArviZ format
        data = pyro_to_arviz(samples)
        self.assertIn("ratings", data.posterior)
        self.assertEqual(data.posterior["ratings"].shape[2], num_elements)

if __name__ == '__main__':
    unittest.main()
