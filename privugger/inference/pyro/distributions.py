"""
Distribution conversion utilities for Pyro backend.
"""
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import RelaxedOneHotCategoricalStraightThrough
from privugger.distributions.continuous import Continuous
from privugger.distributions.discrete import Discrete, Constant, TensorConstant, Categorical, Bernoulli, Binomial, DiscreteUniform, Geometric

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
    
    # Discrete distributions
    elif privug_dist.__class__.__name__ == "Categorical":
        precision = 1

        # build logits from the prior probs
        p = torch.tensor(privug_dist.p, dtype=torch.float32)
        logits = torch.log(p)

        # temperature tied to user‐supplied precision
        temperature = max(precision/100.0, 1e-2)

        # construct a straight‐through Gumbel‐Softmax
        base = RelaxedOneHotCategoricalStraightThrough(
            temperature=temperature,
            logits=logits
        )
        # expand into the full batch + category shape
        base = base.expand(dist_shape + (len(p),))
        if is_multi_element:
            base = base.to_event(1)

        # sample the one‐hot
        one_hot = pyro.sample(name, base)

        return one_hot.argmax(-1)

    
    elif privug_dist.__class__.__name__ == "Bernoulli":
        # Convert probability to tensor
        probs = torch.tensor(privug_dist.p, dtype=torch.float32)
        dist_obj = dist.Bernoulli(probs=probs).expand(dist_shape)
        if is_multi_element:
            dist_obj = dist_obj.to_event(1)
        return pyro.sample(name, dist_obj)
    
    elif privug_dist.__class__.__name__ == "Binomial":
        # Convert parameters to tensors
        total_count = torch.tensor(privug_dist.n, dtype=torch.float32)
        probs = torch.tensor(privug_dist.p, dtype=torch.float32)
        dist_obj = dist.Binomial(total_count=total_count, probs=probs).expand(dist_shape)
        if is_multi_element:
            dist_obj = dist_obj.to_event(1)
        return pyro.sample(name, dist_obj)
    
    elif privug_dist.__class__.__name__ == "DiscreteUniform":
        # In Pyro, we can use Categorical with uniform probabilities to represent DiscreteUniform
        # Create a range of values and uniform probabilities
        low = int(privug_dist.lower)
        high = int(privug_dist.upper) + 1  # +1 because upper is inclusive
        num_values = high - low
        
        # Create uniform probabilities
        probs = torch.ones(num_values, dtype=torch.float32) / num_values
        
        # Use Categorical distribution
        cat_dist = dist.Categorical(probs=probs).expand(dist_shape)
        
        # Map categorical outcomes to the actual range
        # This is a transformation: cat_sample + low gives the desired range
        if is_multi_element:
            cat_dist = cat_dist.to_event(1)
        
        # Sample from categorical and transform
        cat_sample = pyro.sample(f"{name}_categorical", cat_dist)
        transformed_sample = cat_sample + low
        
        # Return as a deterministic node with the original name
        return pyro.deterministic(name, transformed_sample)
    
    elif privug_dist.__class__.__name__ == "Geometric":
        # Convert probability to tensor
        probs = torch.tensor(privug_dist.p, dtype=torch.float32)
        dist_obj = dist.Geometric(probs=probs).expand(dist_shape)
        if is_multi_element:
            dist_obj = dist_obj.to_event(1)
        return pyro.sample(name, dist_obj)
    
    elif privug_dist.__class__.__name__ == "Constant" or privug_dist.__class__.__name__ == "TensorConstant":
        # For constants, use a Delta distribution
        value = torch.tensor(privug_dist.val, dtype=torch.float32)
        if isinstance(value, (list, tuple)):
            value = torch.tensor(value, dtype=torch.float32)
        
        # If multi-element, expand to the right shape
        if is_multi_element and not torch.is_tensor(value):
            value = value.expand(dist_shape)
            
        return pyro.deterministic(name, value)
    
    else:
        raise ValueError(f"Unsupported distribution type: {privug_dist.__class__.__name__}")
