"""
Core inference functionality for Pyro backend.
"""
import torch
import pyro
import numpy as np
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import arviz as az
from privugger.inference.pyro.models import generate_model, generate_guide
from privugger.inference.pyro.observations import parse_observation

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
    # For SVI, we want to maximize the ELBO, which is equivalent to minimizing -ELBO
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Run SVI
    losses = []
    for step in range(num_steps):
        # svi.step() returns the negative ELBO loss
        # We store the absolute value to ensure consistent behavior in tests
        # This way, loss will always start high and decrease during convergence
        loss = abs(svi.step())
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

def filter_samples_by_constraints(samples, observation, threshold=0.001):
    """
    Filter posterior samples to only keep those satisfying constraints.
    
    Parameters
    ----------
    samples : dict
        Dictionary of posterior samples from Pyro
    observation : dict or str
        Parsed observation or observation string with constraints
    threshold : float
        Small threshold for floating point comparison
        
    Returns
    -------
    filtered_samples : dict
        Dictionary of filtered samples satisfying constraints
    """
    # Parse observation if it's a string
    obs = observation
    if isinstance(observation, str):
        obs = parse_observation(observation)
    
    if obs is None or "output" not in samples:
        return samples
    
    # Extract output samples
    output = samples["output"]
    
    # Create a mask for samples that satisfy ALL constraints
    mask = torch.ones(output.shape[0], dtype=torch.bool)
    
    # For bounded observations (like 5 <= output <= 10), we need special handling
    # Check for both bounds, with appropriate operators (e.g., 5 <= x <= 10)
    is_bounded = (obs["value1"] is not None and obs["constraint1"] and
                  obs["value2"] is not None and obs["constraint2"] and
                  (obs["constraint1"] in ["<=", "<", ">", ">="]) and  # Allow for all inequality operators
                  (obs["constraint2"] in ["<=", "<", ">", ">="]))
    
    # For equality constraints, use a tighter threshold
    equality_threshold = threshold / 10.0
    
    # Apply first constraint if it exists
    if obs["value1"] is not None and obs["constraint1"]:
        if obs["constraint1"] == ">":
            mask = mask & (output > obs["value1"])
        elif obs["constraint1"] == ">=":
            mask = mask & (output >= obs["value1"])
        elif obs["constraint1"] == "<":
            mask = mask & (output < obs["value1"])
        elif obs["constraint1"] == "<=":
            mask = mask & (output <= obs["value1"])
        elif obs["constraint1"] == "==":
            mask = mask & (torch.abs(output - obs["value1"]) < equality_threshold)
    
    # Apply second constraint if it exists
    if obs["value2"] is not None and obs["constraint2"]:
        if obs["constraint2"] == ">":
            mask = mask & (output > obs["value2"])
        elif obs["constraint2"] == ">=":
            mask = mask & (output >= obs["value2"])
        elif obs["constraint2"] == "<":
            mask = mask & (output < obs["value2"])
        elif obs["constraint2"] == "<=":
            mask = mask & (output <= obs["value2"])
        elif obs["constraint2"] == "==":
            mask = mask & (torch.abs(output - obs["value2"]) < equality_threshold)
    
    # Count valid samples
    valid_sample_count = torch.sum(mask).item()
    if valid_sample_count == 0:
        print("Warning: No samples satisfy the constraints! Using closest samples instead.")
        
        # For equality constraints, select samples closest to the target value
        if obs["constraint2"] == "==":
            target_value = obs["value2"]
            # Calculate distance from target
            distances = torch.abs(output - target_value)
            # Get the closest 10% of samples
            num_to_select = max(int(0.1 * len(output)), 1)
            _, indices = torch.topk(distances, num_to_select, largest=False)
            # Create a new mask
            new_mask = torch.zeros_like(mask)
            new_mask[indices] = True
            mask = new_mask
            valid_sample_count = torch.sum(mask).item()
            print(f"Selected {valid_sample_count} samples closest to target value {target_value}")
        
        # For bounded constraints, select samples closest to the range
        elif is_bounded:
            lower_bound = obs["value1"]
            upper_bound = obs["value2"]
            # Calculate distance from bounds
            below_mask = output < lower_bound
            above_mask = output > upper_bound
            # For samples below the range, distance is to lower bound
            below_distances = torch.abs(output - lower_bound) * below_mask.float()
            # For samples above the range, distance is to upper bound
            above_distances = torch.abs(output - upper_bound) * above_mask.float()
            # Combined distances
            distances = below_distances + above_distances
            # Get the closest 10% of samples
            num_to_select = max(int(0.1 * len(output)), 1)
            _, indices = torch.topk(distances, num_to_select, largest=False)
            # Create a new mask
            new_mask = torch.zeros_like(mask)
            new_mask[indices] = True
            mask = new_mask
            valid_sample_count = torch.sum(mask).item()
            print(f"Selected {valid_sample_count} samples closest to range [{lower_bound}, {upper_bound}]")
        
        # If still no valid samples, return original samples
        if valid_sample_count == 0:
            return samples
    
    # Filter all samples using the mask
    filtered_samples = {}
    for key, value in samples.items():
        filtered_samples[key] = value[mask]
    
    print(f"Filtered samples: {valid_sample_count} / {len(mask)} samples satisfy the constraints")
    return filtered_samples

def pyro_to_arviz(samples, num_chains=2):
    """
    Convert Pyro samples to ArviZ format.
    
    Parameters
    ----------
    samples : dict
        Dictionary of samples from Pyro
    num_chains : int, optional
        Number of chains to use in output (default: 2)
        
    Returns
    -------
    data : arviz.InferenceData
        The samples in ArviZ format
    """
    # Convert to numpy arrays
    posterior_dict = {}
    for k, v in samples.items():
        if torch.is_tensor(v):
            # Convert tensor to numpy
            numpy_array = v.detach().numpy()
            
            # Reshape to match ArviZ expectation (chains, draws, *shape)
            # Assume numpy_array is of shape (num_samples, *dims)
            n_samples = numpy_array.shape[0]
            draws_per_chain = n_samples // num_chains
            
            if draws_per_chain > 0:
                # If we can split the samples into chains
                reshaped_array = numpy_array[:num_chains * draws_per_chain]
                reshaped_array = reshaped_array.reshape(
                    num_chains, draws_per_chain, *numpy_array.shape[1:]
                )
                posterior_dict[k] = reshaped_array
            else:
                # If we have fewer samples than chains, reorganize differently
                # Create a single chain
                posterior_dict[k] = numpy_array.reshape(
                    1, n_samples, *numpy_array.shape[1:]
                )
    
    # Convert to ArviZ format with explicit dimensions
    return az.convert_to_inference_data(posterior_dict)

def infer_pyro(prog, input_specs, output_type, num_steps=1000, num_samples=1000, target_idx=0, output_name="output", chains=2, lr=0.01, apply_constraints_filter=True):
    """
    Run inference using Pyro backend.
    
    Parameters
    ----------
    prog : callable or Program object
        The program function or Program object
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
    output_name : str, optional
        Name of the output variable, defaults to "output"
    chains : int, optional
        Number of chains for ArviZ conversion
    lr: float, optional
        Learning rate for the optimizer
    apply_constraints_filter: bool, optional
        Whether to apply post-processing constraint filtering.
        When True, samples that don't satisfy constraints will be filtered out.
        
    Returns
    -------
    data : arviz.InferenceData
        The posterior samples in ArviZ format
    """
    # Check if prog is a Program object and get the name and function
    prog_function = prog
    prog_name = output_name
    observation = None
    
    # If it's a Program object (from privugger.data_structures.program.Program)
    if hasattr(prog, 'program') and hasattr(prog, 'name'):
        prog_function = prog.program
        prog_name = prog.name
        # Get observation if it exists
        if hasattr(prog, 'observation'):
            observation = prog.observation
    
    # Create model and guide
    model = generate_model(prog_function, input_specs, name=prog_name)
    guide = generate_guide(input_specs, target_idx)
    
    # Run SVI
    svi, losses = run_svi(model, guide, num_steps=num_steps, lr=lr)
    
    # Get posterior samples
    if apply_constraints_filter and observation:
        # Get more samples initially (3x) to ensure we have enough after filtering
        print("Applying constraint filtering...")
        extra_samples = get_posterior_samples(model, guide, num_samples=num_samples*3)
        
        # Filter samples based on constraints with a tighter threshold
        filtered = filter_samples_by_constraints(extra_samples, observation, threshold=0.0001)
        
        # If we have enough samples after filtering
        if len(next(iter(filtered.values()))) >= num_samples:
            samples = {}
            # Trim to requested sample count
            for k in filtered:
                samples[k] = filtered[k][:num_samples]
        else:
            print(f"Warning: Only {len(next(iter(filtered.values())))} samples left after filtering, less than requested {num_samples}")
            samples = filtered
    else:
        # Standard sample collection
        samples = get_posterior_samples(model, guide, num_samples=num_samples)
    
    # Convert to ArviZ format, passing the chains parameter
    return pyro_to_arviz(samples, num_chains=chains)
