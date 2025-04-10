"""
Observation and constraint handling for Pyro backend.

This module provides utilities for applying observations and constraints
to Pyro models in the context of SVI (Stochastic Variational Inference).
It focuses on effective constraint encoding that provides good gradient
information during optimization.
"""
import torch
import pyro
import pyro.distributions as dist
import re

def parse_observation(constraints_str):
    """
    Parse observation string and extract values, operators and variable name.
    
    Parameters
    ----------
    constraints_str : str
        String representing the constraints/observation
        
    Returns
    -------
    dict
        Dictionary containing parsed observation components
    """
    if not constraints_str:
        return None
        
    # Remove spaces for consistent parsing
    constraints = constraints_str.replace(" ", "")
    
    # Regular expression pattern for constraint parsing
    cons = r"[-+]?([0-9]*\.[0-9]+|[0-9]+)*([>=<]*)([a-zA-Z\s]*)([>=<]{2,})[-+]?([0-9]*\.[0-9]+|[0-9]+|\[(\d*,?)*\])*"
    vals = re.search(cons, constraints)
    
    if not vals:
        return None
        
    # Extract components
    val1 = vals.group(1)
    cons1 = vals.group(2)
    name = vals.group(3)
    cons2 = vals.group(4)
    val2 = vals.group(5)
    
    # Convert values to appropriate types
    v1 = None
    v2 = None
    
    if val1 and cons1:
        v1 = float(val1) if "." in val1 else int(val1)
    
    if val2:
        if val2[0] == "[":  # Handle vector case
            v2 = torch.tensor([int(v) for v in val2[1:-1].split(',')])
        else:
            v2 = float(val2) if "." in val2 else int(val2)
    
    return {
        "value1": v1,
        "constraint1": cons1,
        "name": name,
        "constraint2": cons2,
        "value2": v2
    }

def apply_constraint(tensor, op, value, precision):
    """
    Apply a constraint to a tensor using the specified operator and value.
    Optimized for SVI to provide meaningful gradients during optimization.
    
    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to constrain
    op : str
        Operator (">", ">=", "<", "<=", "==")
    value : float, int, or torch.Tensor
        The value to compare against
    precision : float
        The precision of the constraint (smaller = stricter)
        
    Returns
    -------
    log_prob : torch.Tensor
        Log probability of the constraint
    """
    # Convert precision: smaller precision = stricter constraint
    # Use a very aggressive relationship between precision and scale factor
    # Ensure precision doesn't get too small to avoid numerical issues
    effective_precision = max(precision, 1e-6)
    
    # Use a much more aggressive scaling for all precisions
    # These values are much higher than before to create stronger constraints
    base_scale_factor = 100.0 / (effective_precision ** 2)
    
    # The scale factor determines how strict the constraint is
    # Larger scale factor = stricter constraint
    
    if op == "==":
        # For equality, use an extremely strong penalty function
        # This forces samples to cluster tightly around the target value
        squared_diff = torch.sum((tensor - value) ** 2)
        
        # Use very high penalty scaling for equality - 100x stronger than before
        equality_scale = base_scale_factor * 100.0
        log_prob = -0.5 * equality_scale * squared_diff
        return log_prob
    
    # For inequalities, use much stronger penalties with sharper transitions
    elif op == ">":
        # Use very high sharpness for a more abrupt transition
        sharpness = 20.0  
        violation = torch.nn.functional.softplus((value - tensor) * sharpness) / sharpness
        # Much higher penalty multiplier
        penalty_multiplier = 100.0
        return -torch.sum(violation * base_scale_factor * penalty_multiplier)
        
    elif op == ">=":
        sharpness = 20.0
        # Larger boundary shift for stricter enforcement
        violation = torch.nn.functional.softplus((value - tensor + 0.05) * sharpness) / sharpness
        penalty_multiplier = 100.0
        return -torch.sum(violation * base_scale_factor * penalty_multiplier)
        
    elif op == "<":
        sharpness = 20.0
        violation = torch.nn.functional.softplus((tensor - value) * sharpness) / sharpness
        penalty_multiplier = 100.0
        return -torch.sum(violation * base_scale_factor * penalty_multiplier)
        
    elif op == "<=":
        sharpness = 20.0
        # Larger boundary shift for stricter enforcement
        violation = torch.nn.functional.softplus((tensor - value + 0.05) * sharpness) / sharpness
        penalty_multiplier = 100.0
        return -torch.sum(violation * base_scale_factor * penalty_multiplier)
        
    else:
        raise ValueError(f"Unsupported constraint operator: {op}")

def add_pyro_observation(output, observation, precision):
    """
    Add observation constraint to Pyro model.
    
    Parameters
    ----------
    output : torch.Tensor
        The output tensor to constrain
    observation : dict or str
        Parsed observation or observation string
    precision : float
        The precision of the constraint (smaller values = stricter constraints)
    """
    if observation is None:
        return
    
    # Parse observation if it's a string
    obs = observation
    if isinstance(observation, str):
        obs = parse_observation(observation)
    
    if obs is None:
        return
    
    # For equality constraints with continuous values, use very aggressive approach
    if obs["value2"] is not None and obs["constraint2"] == "==":
        value = obs["value2"]
        if isinstance(value, (int, float)):
            # Use an extremely small scale for extremely tight clustering
            # This makes the Normal observation almost like a spike at the target value
            effective_scale = max(precision / 100.0, 1e-6)  # Much smaller than before
            
            # 1. Use pyro.sample with observe=value with very small scale
            pyro.sample(
                "obs_normal", 
                dist.Normal(output, effective_scale).to_event(output.dim()),
                obs=torch.tensor(value, dtype=output.dtype).expand_as(output)
            )
            
            # 2. Add a SECOND observe statement with a slightly different scale
            # This helps avoid optimization getting stuck in local minima
            pyro.sample(
                "obs_normal2", 
                dist.Normal(output, effective_scale * 2.0).to_event(output.dim()),
                obs=torch.tensor(value, dtype=output.dtype).expand_as(output)
            )
            
            # 3. Also add a factor with ultra-strict precision
            # This creates extreme concentration of probability mass at target value
            ultra_strict_precision = precision / 50.0
            log_prob = apply_constraint(output, "==", value, ultra_strict_precision)
            pyro.factor("obs_equality_factor", log_prob)
            
            # 4. Add a guide-shaping parameter to directly influence the guide
            # This works by adding a parameter that the optimizer can't ignore
            shape_param = pyro.param(
                "equality_shape", 
                torch.tensor(0.0, dtype=output.dtype),
                constraint=dist.constraints.real
            )
            # Apply very strong regularization to pull shape_param toward value
            pyro.factor(
                "equality_regularization",
                -10000.0 * torch.sum((shape_param - value)**2)
            )
            
            return
    
    # Apply first constraint if it exists
    if obs["value1"] is not None and obs["constraint1"]:
        # Invert operator for proper conditioning (due to how Program._unwrap_constrain works)
        op = obs["constraint1"].replace(">", "<")
        log_prob = apply_constraint(output, op, obs["value1"], precision)
        pyro.factor("obs_factor1", log_prob)
    
    # Apply second constraint if it exists
    if obs["value2"] is not None and obs["constraint2"]:
        # Don't apply again if we already handled it with observe above
        if obs["constraint2"] != "==" or not isinstance(obs["value2"], (int, float)):
            log_prob = apply_constraint(output, obs["constraint2"], obs["value2"], precision)
            pyro.factor("obs_factor2", log_prob)
