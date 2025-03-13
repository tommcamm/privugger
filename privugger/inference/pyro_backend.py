"""
Stochastic Variable Inference backend for Privugger using Pyro.
"""
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
from pyro.infer import Predictive
import arviz as az
from privugger.distributions.continuous import Continuous
from privugger.distributions.discrete import Discrete, Constant, TensorConstant

def dist_to_pyro(privug_dist, name, hypers=None):
    """
    Convert a Privugger distribution to a Pyro distribution.
    
    Parameters
    ----------
    privug_dist : Privugger distribution
        The Privugger distribution to convert
    name : str
        The name to use for the Pyro distribution
    hypers : list, optional
        List of hyper parameters for the distribution
        
    Returns
    -------
    pyro_dist : Pyro distribution
        The corresponding Pyro distribution
    """
    if isinstance(privug_dist, str):
        # Handle special case for concatenated/stacked distributions
        return None
    
    dist_shape = None if privug_dist.num_elements == -1 else torch.Size([privug_dist.num_elements])
    
    if privug_dist.__class__.__name__ == "Uniform":
        lower = torch.tensor(privug_dist.lower, dtype=torch.float32)
        upper = torch.tensor(privug_dist.upper, dtype=torch.float32)
        return pyro.sample(name, dist.Uniform(lower, upper).expand(dist_shape))
    
    elif privug_dist.__class__.__name__ == "Normal":
        mu = torch.tensor(privug_dist.mu, dtype=torch.float32)
        std = torch.tensor(privug_dist.std, dtype=torch.float32)
        return pyro.sample(name, dist.Normal(mu, std).expand(dist_shape))
    
    elif privug_dist.__class__.__name__ == "Exponential":
        rate = torch.tensor(privug_dist.lam, dtype=torch.float32)
        return pyro.sample(name, dist.Exponential(rate).expand(dist_shape))
    
    elif privug_dist.__class__.__name__ == "Beta":
        alpha = torch.tensor(privug_dist.alpha, dtype=torch.float32)
        beta = torch.tensor(privug_dist.beta, dtype=torch.float32)
        return pyro.sample(name, dist.Beta(alpha, beta).expand(dist_shape))
    
    else:
        raise ValueError(f"Unsupported distribution type: {privug_dist.__class__.__name__}")

def generate_model(prog, input_specs, name="model"):
    """
    Generate a Pyro model function from a Privugger program.
    
    Parameters
    ----------
    prog : callable or None
        The program function to execute, or None if using priors directly
    input_specs : list
        List of input specifications (distributions)
    name : str, optional
        Name for the model
        
    Returns
    -------
    model : callable
        The Pyro model function
    """
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
        if prog is not None:
            # Execute the program with the sampled priors
            result = prog(*prior_samples)
            
            # Return the program output as a deterministic node
            return pyro.deterministic(name, result)
        else:
            # If no program, return the last prior sample (used in concatenation/stacking scenarios)
            return prior_samples[-1] if prior_samples else None
    
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
                
                # Initialize parameters based on the prior type
                if prior.__class__.__name__ == "Uniform":
                    # For Uniform, we initialize at the midpoint with small std
                    loc_init = (prior.lower + prior.upper) / 2.0
                    scale_init = (prior.upper - prior.lower) / 10.0
                    
                    # Define learnable parameters for the target
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
                    pyro.sample(dist_name, dist.Normal(mu_param, sigma_param).expand(
                        torch.Size([prior.num_elements]) if prior.num_elements != -1 else torch.Size([])
                    ))
                
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
                    pyro.sample(dist_name, dist.Normal(mu_param, sigma_param).expand(
                        torch.Size([prior.num_elements]) if prior.num_elements != -1 else torch.Size([])
                    ))
                
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
                    pyro.sample(dist_name, dist.Normal(mu_param, sigma_param).expand(
                        torch.Size([prior.num_elements]) if prior.num_elements != -1 else torch.Size([])
                    ))
    
    return guide_fn

def run_svi(model, guide, num_steps=1000, lr=0.01):
    """
    Run stochastic variational inference.
    
    Parameters
    ----------
    model : callable
        The Pyro model function
    guide : callable
        The Pyro guide function
    num_steps : int, optional
        Number of optimization steps
    lr : float, optional
        Learning rate for the optimizer
        
    Returns
    -------
    svi : SVI
        The SVI object
    losses : list
        List of loss values during optimization
    """
    # Clear the param store in case we're running multiple inferences
    pyro.clear_param_store()
    
    # Set up the optimizer
    optimizer = Adam({"lr": lr})
    
    # Set up SVI with Trace_ELBO loss
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Run SVI
    losses = []
    for step in range(num_steps):
        loss = svi.step()
        losses.append(loss)
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}/{num_steps} - Loss: {loss:.4f}")
    
    return svi, losses

def get_posterior_samples(model, guide, num_samples=1000):
    """
    Get posterior samples using the trained guide.
    
    Parameters
    ----------
    model : callable
        The Pyro model function
    guide : callable
        The trained Pyro guide function
    num_samples : int, optional
        Number of samples to draw
        
    Returns
    -------
    samples : dict
        Dictionary of posterior samples
    """
    # Create a predictive distribution using the guide
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    
    # Get samples
    samples = predictive()
    
    return samples

def pyro_to_arviz(samples):
    """
    Convert Pyro samples to ArviZ format.
    
    Parameters
    ----------
    samples : dict
        Dictionary of samples from Pyro
        
    Returns
    -------
    data : arviz.InferenceData
        The samples in ArviZ format
    """
    # Convert to numpy arrays
    posterior_dict = {}
    for k, v in samples.items():
        if torch.is_tensor(v):
            posterior_dict[k] = v.detach().numpy()
    
    # Convert to ArviZ format
    return az.convert_to_inference_data(posterior_dict)

def infer_pyro(prog, input_specs, output_type, num_steps=1000, num_samples=1000, target_idx=0):
    """
    Run inference using Pyro backend.
    
    Parameters
    ----------
    prog : callable
        The program function
    input_specs : list
        List of input specifications (distributions)
    output_type : type
        Output type of the program
    num_steps : int, optional
        Number of SVI steps
    num_samples : int, optional
        Number of posterior samples
    target_idx : int, optional
        Index of the target individual's distribution
        
    Returns
    -------
    data : arviz.InferenceData
        The posterior samples in ArviZ format
    """
    # Create model and guide
    model = generate_model(prog, input_specs)
    guide = generate_guide(input_specs, target_idx)
    
    # Run SVI
    svi, losses = run_svi(model, guide, num_steps=num_steps)
    
    # Get posterior samples
    samples = get_posterior_samples(model, guide, num_samples=num_samples)
    
    # Convert to ArviZ format
    return pyro_to_arviz(samples)
