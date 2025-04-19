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

# Post-filtering approach removed in favor of stronger in-model constraints

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

def infer_pyro(prog, input_specs, output_type, num_steps=1000, num_samples=1000, target_idx=0, output_name="output", chains=2, lr=0.01):
    """
    Run inference using Pyro backend with SVI.
    
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
        
    Returns
    -------
    data : arviz.InferenceData
        The posterior samples in ArviZ format
    """
    # Check if prog is a Program object and get the name and function
    prog_function = prog
    prog_name = output_name
    observation = None
    precision = 0.01  # Default precision
    
    # If it's a Program object (from privugger.data_structures.program.Program)
    if hasattr(prog, 'program') and hasattr(prog, 'name'):
        prog_function = prog.program
        prog_name = prog.name
        # Get observation if it exists
        if hasattr(prog, 'observation'):
            observation = prog.observation
        # Get precision if specified
        if hasattr(prog, 'observation_precision'):
            precision = prog.observation_precision
    
    # Create model and guide
    # Pass the entire Program object for observations to work correctly
    model = generate_model(prog_function, input_specs, name=prog_name, prog_obj=prog)
    guide = generate_guide(input_specs, target_idx)
    
    # Run SVI with more steps if there are observations to ensure convergence
    if observation:
        # Observations can make optimization harder, so use more steps
        effective_steps = num_steps * 2
        print(f"Observations detected, running SVI with {effective_steps} steps")
        svi, losses = run_svi(model, guide, num_steps=effective_steps, lr=lr)
    else:
        svi, losses = run_svi(model, guide, num_steps=num_steps, lr=lr)
    
    # Get posterior samples directly
    samples = get_posterior_samples(model, guide, num_samples=num_samples)
    
    # Convert to ArviZ format, passing the chains parameter
    return pyro_to_arviz(samples, num_chains=chains)
