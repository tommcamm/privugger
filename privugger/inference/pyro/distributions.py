"""
Distribution conversion utilities for Pyro backend.
"""
import torch
import pyro
import pyro.distributions as dist
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
    
    dist_shape = torch.Size([]) if privug_dist.num_elements == -1 else torch.Size([privug_dist.num_elements])
    
    # Flag to determine if we need to mark distribution as an event
    is_multi_element = privug_dist.num_elements > 1
    
    if privug_dist.__class__.__name__ == "Uniform":
        lower = torch.tensor(privug_dist.lower, dtype=torch.float32)
        upper = torch.tensor(privug_dist.upper, dtype=torch.float32)
        dist_obj = dist.Uniform(lower, upper).expand(dist_shape)
        if is_multi_element:
            dist_obj = dist_obj.to_event(1)
        return pyro.sample(name, dist_obj)
    
    elif privug_dist.__class__.__name__ == "Normal":
        mu = torch.tensor(privug_dist.mu, dtype=torch.float32)
        std = torch.tensor(privug_dist.std, dtype=torch.float32)
        dist_obj = dist.Normal(mu, std).expand(dist_shape)
        if is_multi_element:
            dist_obj = dist_obj.to_event(1)
        return pyro.sample(name, dist_obj)
    
    elif privug_dist.__class__.__name__ == "Exponential":
        rate = torch.tensor(privug_dist.lam, dtype=torch.float32)
        dist_obj = dist.Exponential(rate).expand(dist_shape)
        if is_multi_element:
            dist_obj = dist_obj.to_event(1)
        return pyro.sample(name, dist_obj)
    
    elif privug_dist.__class__.__name__ == "Beta":
        alpha = torch.tensor(privug_dist.alpha, dtype=torch.float32)
        beta = torch.tensor(privug_dist.beta, dtype=torch.float32)
        dist_obj = dist.Beta(alpha, beta).expand(dist_shape)
        if is_multi_element:
            dist_obj = dist_obj.to_event(1)
        return pyro.sample(name, dist_obj)
    
    else:
        raise ValueError(f"Unsupported distribution type: {privug_dist.__class__.__name__}")
