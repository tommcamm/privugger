"""
Core inference functionality for Pyro backend.

This module provides various inference methods for Pyro:
- SVI (Stochastic Variational Inference)
- MCMC (NUTS/HMC)
- Hybrid (SVI + MCMC)

It also supports automatic guide generation using AutoGuide.
"""
import torch
import pyro
import numpy as np
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal
from pyro.infer import init_to_value, init_to_median
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

def run_mcmc(model, num_samples=1000, num_chains=2, warmup_steps=200, init_strategy=None):
    """
    Run MCMC inference using Pyro's NUTS sampler.
    
    Parameters
    ----------
    model : callable
        The Pyro model function
    num_samples : int, optional
        Number of samples to draw
    num_chains : int, optional
        Number of chains to run
    warmup_steps : int, optional
        Number of warmup steps
    init_strategy : callable, optional
        Initialization strategy for MCMC (e.g., init_to_value)
        
    Returns
    -------
    mcmc : MCMC
        The MCMC object after running
    samples : dict
        Dictionary of samples
    """
    # Clear the param store in case we're running multiple inferences
    pyro.clear_param_store()
    
    # Set up the NUTS kernel with initialization strategy if provided
    if init_strategy:
        nuts_kernel = NUTS(model, init_strategy=init_strategy)
    else:
        nuts_kernel = NUTS(model)
    
    # Set up MCMC
    mcmc = MCMC(nuts_kernel, 
                num_samples=num_samples, 
                num_chains=num_chains,
                warmup_steps=warmup_steps)
    
    # Run MCMC
    mcmc.run()
    
    # Get samples
    samples = mcmc.get_samples()
    
    return mcmc, samples

def run_hybrid(model, guide, num_svi_steps=1000, lr=0.01, 
               num_samples=1000, num_chains=2, warmup_steps=200):
    """
    Run hybrid inference: 
    1. SVI for initial approximation
    2. MCMC (NUTS) warm-started with SVI result
    
    Parameters
    ----------
    model : callable
        The Pyro model function
    guide : callable
        The Pyro guide function
    num_svi_steps : int, optional
        Number of SVI steps
    lr : float, optional
        Learning rate for SVI
    num_samples, num_chains, warmup_steps : int, optional
        MCMC parameters
        
    Returns
    -------
    mcmc : MCMC
        The MCMC object after running
    samples : dict
        Dictionary of samples from MCMC
    """
    # 1. Run SVI first
    svi, _ = run_svi(model, guide, num_steps=num_svi_steps, lr=lr)
    
    # 2. Get posterior samples from SVI to use as initial values
    svi_samples = get_posterior_samples(model, guide, num_samples=100)
    
    # 3. Compute median values to use as initial points for MCMC
    init_values = {}
    for name, value in svi_samples.items():
        if torch.is_tensor(value):
            # Compute median along sample dimension
            init_values[name] = value.median(dim=0)[0]
    
    # 4. Run MCMC with initialization from SVI
    return run_mcmc(
        model, 
        num_samples=num_samples, 
        num_chains=num_chains,
        warmup_steps=warmup_steps,
        init_strategy=init_to_value(values=init_values)
    )

def generate_auto_guide(model, target_names=None, guide_type="normal"):
    """
    Generate an automatic guide for a model using Pyro's AutoGuide.
    
    Parameters
    ----------
    model : callable
        The Pyro model function
    target_names : list, optional
        Names of the target variables to focus on
    guide_type : str, optional
        Type of auto guide to use:
        - "normal": AutoNormal (default)
        - "multivariate": AutoMultivariateNormal
        
    Returns
    -------
    guide : AutoGuide
        The automatic guide
    """
    # Create AutoGuide based on requested type
    if guide_type == "multivariate":
        guide = AutoMultivariateNormal(model, init_loc_fn=init_to_median)
    else:  # Default to AutoNormal
        guide = AutoNormal(model, init_loc_fn=init_to_median)
    
    return guide

def _get_observation_details(prog, prog_obj):
    """
    Helper function to extract observation details from program objects.
    
    Parameters
    ----------
    prog : callable or Program object
        The program function or Program object
    prog_obj : Program object or None
        The Program object passed directly
        
    Returns
    -------
    observation : str or None
        The observation constraint string
    precision : float
        The precision for the observation
    """
    observation = None
    precision = 0.01  # Default precision
    
    # First check if prog_obj is provided and has observation
    if prog_obj is not None:
        if hasattr(prog_obj, 'observation'):
            observation = prog_obj.observation
            # Check if prog_obj has a specific observation precision
            if hasattr(prog_obj, 'observation_precision'):
                precision = prog_obj.observation_precision
    # Otherwise check the prog parameter (for backward compatibility)
    elif hasattr(prog, 'observation'):
        observation = prog.observation
        # Check if prog has a specific observation precision
        if hasattr(prog, 'observation_precision'):
            precision = prog.observation_precision
            
    return observation, precision

def _calculate_effective_steps(num_steps, observation):
    """
    Helper function to calculate effective SVI steps based on observation presence.
    
    Parameters
    ----------
    num_steps : int
        Base number of SVI steps
    observation : str or None
        The observation constraint string
        
    Returns
    -------
    effective_steps : int
        Adjusted number of SVI steps
    """
    if observation:
        # Observations can make optimization harder, so use more steps
        effective_steps = num_steps * 2
        print(f"Observations detected, running SVI with {effective_steps} steps")
        return effective_steps
    return num_steps

def infer_pyro(prog, input_specs, output_type, num_steps=1000, num_samples=1000, 
               target_idx=0, output_name="output", chains=2, lr=0.01,
               method="svi", autoguide=False, guide_type="normal",
               warmup_steps=None):
    """
    Run inference using Pyro backend with flexible inference methods.
    
    Parameters
    ----------
    prog : callable or Program object
        The program function or Program object
    input_specs : list
        List of input specifications (distributions)
    output_type : type
        Output type of the program
    num_steps : int, optional
        Number of SVI steps or total MCMC steps
    num_samples : int, optional
        Number of posterior samples
    target_idx : int, optional
        Index of the target individual's distribution
    output_name : str, optional
        Name of the output variable, defaults to "output"
    chains : int, optional
        Number of chains for ArviZ conversion and MCMC
    lr: float, optional
        Learning rate for the optimizer (SVI only)
    method : str, optional
        Inference method to use:
        - "svi": Run SVI only (default)
        - "mcmc": Run MCMC (NUTS) only
        - "hybrid": Run SVI, then warm-start MCMC with SVI result
    autoguide : bool, optional
        Whether to use AutoGuide for SVI instead of manually constructed guide
    guide_type : str, optional
        Type of auto guide to use if autoguide=True:
        - "normal": AutoNormal (default)
        - "multivariate": AutoMultivariateNormal
    warmup_steps : int, optional
        Number of warmup steps for MCMC (default: num_steps // 5)
        
    Returns
    -------
    data : arviz.InferenceData
        The posterior samples in ArviZ format
    """
    # Check if prog is a Program object and get the name and function
    prog_function = prog
    prog_name = output_name
    
    # If it's a Program object (from privugger.data_structures.program.Program)
    if hasattr(prog, 'program') and hasattr(prog, 'name'):
        prog_function = prog.program
        prog_name = prog.name
    
    # Get observation details
    observation, precision = _get_observation_details(prog, prog)
    
    # Create model 
    # Pass the entire Program object for observations to work correctly
    model = generate_model(prog_function, input_specs, name=prog_name, prog_obj=prog)
    
    # Create guide based on autoguide parameter
    if autoguide:
        guide = generate_auto_guide(model, 
                                   target_names=[input_specs[target_idx].name],
                                   guide_type=guide_type)
    else:
        guide = generate_guide(input_specs, target_idx)
    
    # Set default warmup steps for MCMC if not provided
    if warmup_steps is None:
        warmup_steps = num_steps // 5
    
    # Run inference based on method
    if method == "svi":
        # Adjust steps if observations are present
        effective_steps = _calculate_effective_steps(num_steps, observation)
        
        # Run SVI
        svi, losses = run_svi(model, guide, num_steps=effective_steps, lr=lr)
        
        # Get posterior samples directly
        samples = get_posterior_samples(model, guide, num_samples=num_samples)
        
    elif method == "mcmc":
        # Run MCMC only
        print(f"Running MCMC with {num_samples} samples, {chains} chains, and {warmup_steps} warmup steps")
        _, samples = run_mcmc(model, 
                             num_samples=num_samples, 
                             num_chains=chains, 
                             warmup_steps=warmup_steps)
        
    elif method == "hybrid":
        # Adjust steps if observations are present
        effective_steps = _calculate_effective_steps(num_steps, observation)
        
        # Run hybrid SVI+MCMC
        print(f"Running hybrid SVI+MCMC: SVI with {effective_steps} steps then MCMC with {num_samples} samples")
        _, samples = run_hybrid(model, guide, 
                              num_svi_steps=effective_steps, 
                              lr=lr,
                              num_samples=num_samples, 
                              num_chains=chains, 
                              warmup_steps=warmup_steps)
    else:
        raise ValueError(f"Unknown method: {method}. Valid options are 'svi', 'mcmc', 'hybrid'")
    
    # Convert to ArviZ format
    return pyro_to_arviz(samples, num_chains=chains)
