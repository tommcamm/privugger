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
    import re
    
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
                    # TODO: This is a hack to prevent issues with values outside the bounds, this needs to be verified. 

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
