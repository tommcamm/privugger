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
from privugger.inference.pyro_backend import dist_to_pyro, generate_model, generate_guide, run_svi, parse_observation, apply_constraint

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
    
    # Tests for the observation feature
    
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
    
    def test_inequality_observation(self):
        """
        Test Pyro backend with an inequality observation
        """
        # Define a simple prior with a wider range
        age = pv.Normal("age", mu=20.0, std=10.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[age])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add observation: output must be >= 25
        # Use smaller precision value for stricter constraint
        prog.add_observation("output >= 25", precision=0.01)
        
        # Run inference with Pyro backend - use more SVI steps
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005)
        
        # Get output samples
        output_samples = trace.posterior["output"].values.flatten()
        
        # Check that a majority of the samples respect the constraint
        # The constraint is soft, so we use a moderate threshold of 60%
        constrained_samples_ratio = np.mean(output_samples >= 25)
        self.assertGreaterEqual(constrained_samples_ratio, 0.6, 
                              f"Only {constrained_samples_ratio:.2%} of samples respect the constraint")
        
        # The mean should be greater than the prior mean due to the constraint
        self.assertGreater(output_samples.mean(), 20.0)
    
    def test_bounded_observation(self):
        """
        Test Pyro backend with a bounded observation
        """
        # Define a simple prior with a wider range
        x = pv.Normal("x", mu=0.0, std=10.0)
        
        # Create dataset and program
        ds = pv.Dataset(input_specs=[x])
        prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)
        
        # Add bounded observation: 5 <= output <= 10
        # Use smaller precision value for stricter constraint
        prog.add_observation("5 <= output <= 10", precision=0.01)
        
        # Run inference with Pyro backend - more steps and lower learning rate
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005)
        
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
    
    def test_equality_observation(self):
        """
        Test Pyro backend with an equality observation
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
        
        # Run inference with Pyro backend - more steps and lower learning rate
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005)
        
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
            
            # Run inference with Pyro backend
            trace = pv.infer(prog, draws=500, method="pyro", svi_steps=500)
            
            # Get output samples
            output_samples = trace.posterior["output"].values.flatten()
            
            # Calculate proportion of samples that respect the constraint
            constraint_adherence.append(np.mean(output_samples >= 3))
        
        # Lower precision (stricter) should have higher constraint adherence
        # Higher precision (more lenient) should have lower constraint adherence
        self.assertGreaterEqual(constraint_adherence[0], constraint_adherence[2],
                              "Stricter precision should lead to higher constraint adherence")
    
    def test_observation_with_addition(self):
        """
        Test observation with addition program
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
        
        # Run inference with Pyro backend - more steps, lower learning rate
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005)
        
        # Get samples
        output_samples = trace.posterior["output"].values.flatten()
        
        # Check constraint adherence
        constraint_adherence = np.mean(output_samples >= 60)
        self.assertGreaterEqual(constraint_adherence, 0.6, 
                              f"Only {constraint_adherence:.2%} of samples respect the constraint")
        
        # The mean should be greater than the unconstrained mean (10 + 40 = 50)
        self.assertGreater(output_samples.mean(), 50.0)
    
    def test_observation_with_multiplication(self):
        """
        Test observation with multiplication program
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
        
        # Run inference with Pyro backend - more steps, lower learning rate
        trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005)
        
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
