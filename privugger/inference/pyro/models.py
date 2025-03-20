"""
Model and guide generation for Pyro backend.
"""
import torch
import pyro
import pyro.distributions as dist
import importlib.util
import os
from privugger.inference.pyro.distributions import dist_to_pyro
from privugger.inference.pyro.observations import add_pyro_observation

def generate_model(prog, input_specs, name="output"):
    """
    Generate a Pyro model function from a Privugger program.
    
    Parameters
    ----------
    prog : callable, str, or None
        The program function to execute, a string path to a program file, 
        or None if using priors directly
    input_specs : list
        List of input specifications (distributions)
    name : str, optional
        Name for the model
        
    Returns
    -------
    model : callable
        The Pyro model function
    """
    # If prog is a string (file path), load the function from the file
    prog_function = prog
    if isinstance(prog, str):
        import importlib.util
        import os
        
        # Get the absolute path
        file_path = os.path.abspath(prog)
        
        # Load the module
        module_name = os.path.basename(file_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the 'name' function from the module
        prog_function = getattr(module, 'name')
    
    # Extract observation details if prog is a Program object
    observation = None
    precision = 0.01  # Default precision
    
    if hasattr(prog, 'observation'):
        observation = prog.observation
        # Check if prog has a specific observation precision
        if hasattr(prog, 'observation_precision'):
            precision = prog.observation_precision
    
    def model_fn():
        # Sample from priors
        prior_samples = []
        
        for idx, prior in enumerate(input_specs):
            if isinstance(prior, str):
                # Skip string descriptions (used for concatenated/stacked variables)
                continue
            
            if prior.is_hyper_param:
                # Skip hyper parameters, they should be handled within their respective distributions
                continue
            
            # Convert the prior to a Pyro distribution and sample
            prior_sample = dist_to_pyro(prior, prior.name)
            prior_samples.append(prior_sample)
        
        # If there's a program function, execute it with the prior samples
        if prog_function is not None:
            # Execute the program with the sampled priors
            result = prog_function(*prior_samples)
            
            # Add observation constraint if present
            if observation is not None:
                add_pyro_observation(result, observation, precision)
                
            # Return the program output as a deterministic node
            return pyro.deterministic(name, result)
        else:
            output = prior_samples[-1] if prior_samples else None
            
            # Add observation constraint if present
            if observation is not None and output is not None:
                add_pyro_observation(output, observation, precision)
                
            # If no program, return the last prior sample (used in concatenation/stacking scenarios)
            return output
    
    return model_fn

def generate_guide(input_specs, target_idx=0, name="guide"):
    """
    Generate a Pyro guide function for SVI.
    
    Parameters
    ----------
    input_specs : list
        List of input specifications (distributions)
    target_idx : int, optional
        Index of the target individual's distribution
    name : str, optional
        Name for the guide
        
    Returns
    -------
    guide : callable
        The Pyro guide function
    """
    def guide_fn():
        # For each prior, define learnable parameters for the approximate posterior
        for idx, prior in enumerate(input_specs):
            if isinstance(prior, str) or prior.is_hyper_param:
                continue
            
            # Focus on the target individual (usually at target_idx)
            if idx == target_idx:
                dist_name = prior.name
                dist_shape = torch.Size([prior.num_elements]) if prior.num_elements != -1 else torch.Size([])
                
                # Initialize parameters based on the prior type
                if prior.__class__.__name__ == "Uniform":
                    # For Uniform, use a transformed distribution to ensure values stay within bounds
                    lower = torch.tensor(prior.lower, dtype=torch.float32)
                    upper = torch.tensor(prior.upper, dtype=torch.float32)
                    
                    # Initialize parameters in unconstrained space  
                    # Start at the logit of the midpoint to ensure it maps close to the center
                    midpoint = (prior.lower + prior.upper) / 2.0
                    # Map [lower, upper] -> [0, 1] -> unconstrained space
                    normalized_midpoint = (midpoint - prior.lower) / (prior.upper - prior.lower)
                    # Apply logit transformation (inverse of sigmoid) to get to unconstrained space
                    # Add small epsilon to avoid log(0) or log(1)
                    epsilon = 1e-6
                    # Convert to tensor before applying torch.log
                    init_value = torch.log(torch.tensor(normalized_midpoint / (1 - normalized_midpoint + epsilon) + epsilon, dtype=torch.float32))
                    
                    # Create unconstrained parameter
                    loc_param = pyro.param(
                        f"loc_{dist_name}",
                        torch.tensor(init_value, dtype=torch.float32)
                    )
                    
                    # Create a scale parameter in unconstrained space
                    scale_param = pyro.param(
                        f"scale_{dist_name}",
                        torch.tensor(0.1, dtype=torch.float32),
                        constraint=dist.constraints.positive
                    )
                    
                    # Create a Normal distribution in unconstrained space
                    base_dist = dist.Normal(loc_param, scale_param).expand(dist_shape)
                    
                    # Create a sigmoid transformation to map R -> (0,1)
                    sigmoid_transform = dist.transforms.SigmoidTransform()
                    
                    # Create an affine transformation to map (0,1) -> (lower, upper)
                    affine_transform = dist.transforms.AffineTransform(
                        loc=lower,
                        scale=upper - lower
                    )
                    
                    # Compose the transformations
                    transform = dist.transforms.ComposeTransform([sigmoid_transform, affine_transform])
                    
                    # Create a transformed distribution for sampling
                    transformed_dist = dist.TransformedDistribution(base_dist, transform)
                    
                    # Sample from the transformed distribution
                    pyro.sample(dist_name, transformed_dist)
                
                elif prior.__class__.__name__ == "Beta":
                    # For Beta, we can use a Beta distribution directly for the posterior
                    # Initialize with the prior parameters
                    alpha_param = pyro.param(
                        f"alpha_{dist_name}",
                        torch.tensor(prior.alpha, dtype=torch.float32),
                        constraint=dist.constraints.positive
                    )
                    beta_param = pyro.param(
                        f"beta_{dist_name}",
                        torch.tensor(prior.beta, dtype=torch.float32),
                        constraint=dist.constraints.positive
                    )
                    
                    # Sample from Beta distribution directly
                    dist_obj = dist.Beta(alpha_param, beta_param).expand(dist_shape)
                    if prior.num_elements > 1:  # If it's a multi-element distribution
                        dist_obj = dist_obj.to_event(1)
                    pyro.sample(dist_name, dist_obj)
                
                elif prior.__class__.__name__ == "Normal":
                    # For Normal, initialize with the prior's parameters
                    mu_param = pyro.param(
                        f"mu_{dist_name}", 
                        torch.tensor(prior.mu, dtype=torch.float32)
                    )
                    sigma_param = pyro.param(
                        f"sigma_{dist_name}", 
                        torch.tensor(prior.std, dtype=torch.float32),
                        constraint=dist.constraints.positive
                    )
                    
                    # Sample from the approximate posterior
                    dist_obj = dist.Normal(mu_param, sigma_param).expand(dist_shape)
                    if prior.num_elements > 1:  # If it's a multi-element distribution
                        dist_obj = dist_obj.to_event(1)
                    pyro.sample(dist_name, dist_obj)
                
                # Add support for other distribution types as needed
                else:
                    # Default to Normal for other distribution types
                    if hasattr(prior, 'mu') and hasattr(prior, 'std'):
                        loc_init = prior.mu
                        scale_init = prior.std
                    else:
                        # Fallback initialization
                        loc_init = 0.0
                        scale_init = 1.0
                    
                    mu_param = pyro.param(
                        f"mu_{dist_name}", 
                        torch.tensor(loc_init, dtype=torch.float32)
                    )
                    sigma_param = pyro.param(
                        f"sigma_{dist_name}", 
                        torch.tensor(scale_init, dtype=torch.float32),
                        constraint=dist.constraints.positive
                    )
                    
                    # Sample from the approximate posterior
                    dist_obj = dist.Normal(mu_param, sigma_param).expand(dist_shape)
                    if prior.num_elements > 1:  # If it's a multi-element distribution
                        dist_obj = dist_obj.to_event(1)
                    pyro.sample(dist_name, dist_obj)
    
    return guide_fn
