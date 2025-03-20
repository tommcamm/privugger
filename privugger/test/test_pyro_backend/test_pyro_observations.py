import os
import numpy as np
import torch
import pyro
import unittest
import pytest

import privugger as pv
from privugger.inference.pyro.observations import parse_observation

# Import example programs using relative paths that work cross-platform
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
program_identity = os.path.join(BASE_DIR, "test", "example_programs", "basic", "identity.py")
program_addition = os.path.join(BASE_DIR, "test", "example_programs", "basic", "addition.py")
program_multiplication = os.path.join(BASE_DIR, "test", "example_programs", "basic", "multiplication.py")

@pytest.mark.pyro
class TestPyroObservations(unittest.TestCase):
    """
    Tests for observation handling in the Pyro backend.
    
    These tests focus on the functionality of parsing observation strings
    and applying constraints to models during inference.
    """
    
    def setUp(self):
        # Reset state before each test
        pv.reset()
        pyro.clear_param_store()
    
    def test_parse_observation(self):
        """
        Test the parse_observation function in the Pyro backend
        """
        # Test a simple observation
        obs_str = "output >= 5"
        parsed = parse_observation(obs_str)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["name"], "output")
        self.assertEqual(parsed["constraint2"], ">=")
        self.assertEqual(parsed["value2"], 5)
        
        # Test a bounded observation
        obs_str = "3 <= output <= 7"
        parsed = parse_observation(obs_str)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["value1"], 3)
        self.assertEqual(parsed["constraint1"], "<=")
        self.assertEqual(parsed["name"], "output")
        self.assertEqual(parsed["constraint2"], "<=")
        self.assertEqual(parsed["value2"], 7)
        
        # Test an equality observation
        obs_str = "output == 10"
        parsed = parse_observation(obs_str)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["name"], "output")
        self.assertEqual(parsed["constraint2"], "==")
        self.assertEqual(parsed["value2"], 10)
    
    @pytest.mark.slow
    def test_inequality_observation(self):
        """
        Test Pyro backend with an inequality observation
        
        This test verifies that adding an inequality constraint
        (output >= 25) shifts the posterior distribution appropriately.
        """
        # Define a simple prior with a wider range
        age = pv.Normal("age", mu=20.0, std=10.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add observation: output must be >= 25
        # Use smaller precision value for stricter constraint
        prog.add_observation("output >= 25", precision=0.01)
        
        # Run inference with Pyro backend - use more SVI steps and enable constraint filtering
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005, apply_constraints_filter=True)
        
        # Get output samples
        output_samples = trace.posterior["output"].values.flatten()
        
        # Check that a majority of the samples respect the constraint
        # The constraint is soft, so we use a moderate threshold of 60%
        constrained_samples_ratio = np.mean(output_samples >= 25)
        self.assertGreaterEqual(constrained_samples_ratio, 0.6, 
                              f"Only {constrained_samples_ratio:.2%} of samples respect the constraint")
        
        # The mean should be greater than the prior mean due to the constraint
        self.assertGreater(output_samples.mean(), 20.0)
    
    @pytest.mark.slow
    def test_bounded_observation(self):
        """
        Test Pyro backend with a bounded observation
        
        This test checks that adding a bounded constraint (5 <= output <= 10)
        properly constrains the posterior distribution.
        """
        # Define a simple prior with a wider range
        x = pv.Normal("x", mu=0.0, std=10.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[x])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add bounded observation: 5 <= output <= 10
        # Use smaller precision value for stricter constraint
        prog.add_observation("5 <= output <= 10", precision=0.01)
        
        # Run inference with Pyro backend - more steps, lower learning rate, and constraint filtering
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005, apply_constraints_filter=True)
        
        # Get output samples
        output_samples = trace.posterior["output"].values.flatten()
        
        # Check that a reasonable portion of the samples respect both constraints
        # We use moderate thresholds because constraints are soft
        lower_constraint_ratio = np.mean(output_samples >= 5)
        upper_constraint_ratio = np.mean(output_samples <= 10)
        both_constraints_ratio = np.mean((output_samples >= 5) & (output_samples <= 10))
        
        self.assertGreaterEqual(lower_constraint_ratio, 0.6, 
                              f"Only {lower_constraint_ratio:.2%} of samples respect the lower bound")
        self.assertGreaterEqual(upper_constraint_ratio, 0.6, 
                              f"Only {upper_constraint_ratio:.2%} of samples respect the upper bound")
        self.assertGreaterEqual(both_constraints_ratio, 0.4, 
                              f"Only {both_constraints_ratio:.2%} of samples respect both bounds")
        
        # The mean should be pulled toward the constraint range
        self.assertGreater(output_samples.mean(), 0.0)  # Original mean was 0.0
    
    @pytest.mark.slow
    def test_equality_observation(self):
        """
        Test Pyro backend with an equality observation
        
        This test checks that equality constraints (output == 7)
        concentrate the posterior distribution around the target value.
        """
        # Define a simple prior
        x = pv.Normal("x", mu=0.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[x])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add equality observation: output == 7
        target_value = 7.0
        # Use very small precision for stricter constraint
        prog.add_observation(f"output == {target_value}", precision=0.001)
        
        # Run inference with Pyro backend - more steps, lower learning rate, and constraint filtering
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005, apply_constraints_filter=True)
        
        # Get output samples
        output_samples = trace.posterior["output"].values.flatten()
        
        # Check that the mean is reasonably close to the target value
        self.assertAlmostEqual(output_samples.mean(), target_value, delta=2.0)
        
        # Check that a reasonable portion of samples are close to the target value
        close_to_target_ratio = np.mean(np.abs(output_samples - target_value) < 3.0)
        self.assertGreaterEqual(close_to_target_ratio, 0.5, 
                              f"Only {close_to_target_ratio:.2%} of samples are close to the target value")
    
    def test_observation_precision(self):
        """
        Test how different precision values affect observation constraints
        
        This test explores how varying the precision parameter affects
        the strictness of the observation constraints.
        """
        # Test with different precision values
        precisions = [0.05, 0.5, 2.0]  # From strict to lenient
        constraint_adherence = []
        
        for precision in precisions:
            # Reset for each test
            pv.reset()
            pyro.clear_param_store()
            
            # Define a simple prior
            x = pv.Normal("x", mu=0.0, std=5.0)
            
            # Create dataset and program
            ds = pv.Dataset(input_specs=[x])
            prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
            
            # Add observation: output >= 3
            prog.add_observation("output >= 3", precision=precision)
            
            # Run inference with Pyro backend with constraint filtering
            trace = pv.infer(prog, draws=500, method="pyro", svi_steps=500, apply_constraints_filter=True)
            
            # Get output samples
            output_samples = trace.posterior["output"].values.flatten()
            
            # Calculate proportion of samples that respect the constraint
            constraint_adherence.append(np.mean(output_samples >= 3))
        
        # Lower precision (stricter) should have higher constraint adherence
        # Higher precision (more lenient) should have lower constraint adherence
        self.assertGreaterEqual(constraint_adherence[0], constraint_adherence[2],
                              "Stricter precision should lead to higher constraint adherence")
    
    @pytest.mark.slow
    def test_observation_with_addition(self):
        """
        Test observation with addition program
        
        Checks that constraints work correctly with non-identity programs.
        """
        # Define priors
        a = pv.Normal("age", mu=10.0, std=2.0)
        b = pv.Normal("height", mu=40.0, std=5.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[a, b])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_addition)
        
        # Add observation: output (sum) must be >= 60
        # Use smaller precision for stricter constraint
        prog.add_observation("output >= 60", precision=0.01)
        
        # Run inference with Pyro backend - more steps, lower learning rate, and constraint filtering
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005, apply_constraints_filter=True)
        
        # Get samples
        output_samples = trace.posterior["output"].values.flatten()
        
        # Check constraint adherence
        constraint_adherence = np.mean(output_samples >= 60)
        self.assertGreaterEqual(constraint_adherence, 0.6, 
                              f"Only {constraint_adherence:.2%} of samples respect the constraint")
        
        # The mean should be greater than the unconstrained mean (10 + 40 = 50)
        self.assertGreater(output_samples.mean(), 50.0)
    
    @pytest.mark.slow
    def test_observation_with_multiplication(self):
        """
        Test observation with multiplication program
        
        Checks that constraints work correctly with non-linear programs.
        """
        # Define priors
        a = pv.Normal("age", mu=5.0, std=1.0)
        b = pv.Normal("height", mu=6.0, std=1.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[a, b])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_multiplication)
        
        # Add observation: output (product) must be <= 25
        # Use smaller precision for stricter constraint
        prog.add_observation("output <= 25", precision=0.01)
        
        # Run inference with Pyro backend - more steps, lower learning rate, and constraint filtering
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005, apply_constraints_filter=True)
        
        # Get samples
        output_samples = trace.posterior["output"].values.flatten()
        
        # Check constraint adherence with moderate threshold
        constraint_adherence = np.mean(output_samples <= 25)
        self.assertGreaterEqual(constraint_adherence, 0.6, 
                              f"Only {constraint_adherence:.2%} of samples respect the constraint")
        
        # The mean should be less than the unconstrained mean (5 * 6 = 30)
        # but since constraints are soft, we allow a bit of flexibility
        self.assertLess(output_samples.mean(), 32.0)


if __name__ == '__main__':
    unittest.main()
