"""
Observation and constraint handling for Pyro backend.
"""
import torch
import pyro
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
    # This scales the penalty: smaller precision = higher penalty
    scale_factor = 1.0 / (precision ** 2)
    
    # Apply a stronger scale factor for inequality constraints to improve enforcement
    inequality_boost = 10.0
    
    if op == ">":
        # Penalize when tensor <= value
        violation = torch.nn.functional.relu(value - tensor + 1e-6)
        log_prob = -torch.sum(violation * scale_factor * inequality_boost)
    elif op == ">=":
        # Penalize when tensor < value
        # Apply stronger penalty for lower bound violations by using a higher scale factor
        violation = torch.nn.functional.relu(value - tensor)
        log_prob = -torch.sum(violation * scale_factor * inequality_boost)
    elif op == "<":
        # Penalize when tensor >= value
        violation = torch.nn.functional.relu(tensor - value + 1e-6)
        log_prob = -torch.sum(violation * scale_factor * inequality_boost)
    elif op == "<=":
        # Penalize when tensor > value
        violation = torch.nn.functional.relu(tensor - value)
        log_prob = -torch.sum(violation * scale_factor * inequality_boost)
    elif op == "==":
        # Gaussian likelihood centered at value - stronger penalty for equality constraints
        log_prob = -torch.sum((tensor - value) ** 2 * scale_factor)
    else:
        raise ValueError(f"Unsupported constraint operator: {op}")
    
    return log_prob

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
        The precision of the constraint
    """
    if observation is None:
        return
    
    # Parse observation if it's a string
    obs = observation
    if isinstance(observation, str):
        obs = parse_observation(observation)
    
    if obs is None:
        return
    
    # Apply first constraint if it exists
    if obs["value1"] is not None and obs["constraint1"]:
        # Invert operator for proper conditioning (due to how Program._unwrap_constrain works)
        op = obs["constraint1"].replace(">", "<")
        log_prob = apply_constraint(output, op, obs["value1"], precision)
        pyro.factor("obs_factor1", log_prob)
    
    # Apply second constraint if it exists
    if obs["value2"] is not None and obs["constraint2"]:
        log_prob = apply_constraint(output, obs["constraint2"], obs["value2"], precision)
        pyro.factor("obs_factor2", log_prob)
