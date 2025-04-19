# Privugger Pyro Backend Implementation Report

## Introduction

This report explains how the Privugger backend works when using the "Pyro" backend, focusing specifically on the Stochastic Variational Inference (SVI) implementation and how observations are handled in Pyro models. The report also compares the differences between Markov Chain Monte Carlo (MCMC) and SVI approaches for inference.

## Core Architecture

Privugger's Pyro backend is implemented in `privugger/inference/pyro_backend.py` and provides probabilistic programming capabilities using the Pyro framework. The backend enables privacy analysis by modeling programs as probabilistic models where:

1. Inputs are treated as random variables with specified distributions
2. Program execution is modeled as a deterministic transformation of these random variables
3. Inference methods are used to analyze the posterior distribution of variables given constraints

The architecture follows these key components:

- **Distribution conversion**: Maps Privugger's distribution classes to Pyro distributions
- **Model generation**: Creates Pyro models from Privugger programs
- **Guide generation**: Creates variational guides for SVI
- **Inference execution**: Runs the inference using SVI or other methods
- **Observation handling**: Adds observation constraints to models

## Inference Methods

### SVI Implementation

SVI (Stochastic Variational Inference) is the primary inference method implemented in the Pyro backend. It works by:

1. Approximating the true posterior distribution with a simpler, parameterized distribution (the guide)
2. Optimizing the parameters of this guide to minimize the divergence from the true posterior
3. Using stochastic gradient descent for optimization

Key components of Privugger's SVI implementation include:

```python
def run_svi(model, guide, num_steps=1000, lr=0.01):
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
```

The SVI implementation uses:
- **Adam optimizer**: An adaptive learning rate optimization algorithm
- **Trace_ELBO**: Evidence Lower Bound objective function that measures how well the guide approximates the model
- **Iterative optimization**: Gradually improves the guide through multiple steps

### Guide Function Implementation

The guide is crucial for SVI as it defines the form of the approximate posterior. Privugger generates appropriate guides based on the prior distributions:

```python
def generate_guide(input_specs, target_idx=0, name="guide"):
    def guide_fn():
        # For each prior, define learnable parameters for the approximate posterior
        for idx, prior in enumerate(input_specs):
            if isinstance(prior, str) or prior.is_hyper_param:
                continue
            
            # Focus on the target individual (usually at target_idx)
            if idx == target_idx:
                # Guide implementation varies based on distribution type
                if prior.__class__.__name__ == "Uniform":
                    # Use transformed distributions for bounded variables
                    # ...
                elif prior.__class__.__name__ == "Beta":
                    # Use Beta distribution directly
                    # ...
                elif prior.__class__.__name__ == "Normal":
                    # Parameterize with mu and sigma
                    # ...
    return guide_fn
```

The guide implementation is distribution-specific:
- **Normal distributions**: Parameterized directly with learnable mean and standard deviation
- **Uniform distributions**: Uses transformed distributions to respect bounds
- **Beta distributions**: Uses learnable alpha and beta parameters

### MCMC vs. SVI Comparison

While the Pyro backend primarily uses SVI, it's important to understand the tradeoffs compared to MCMC:

| Aspect | SVI (Pyro) | MCMC (e.g., PyMC3) |
|--------|------------|-------------------|
| Speed | Faster, especially for large datasets | Slower, more computationally intensive |
| Scalability | Scales better to high dimensions | Can struggle with many parameters |
| Accuracy | Approximate, depends on guide quality | Asymptotically exact |
| Convergence | Optimization-based, can get stuck | Sampling-based, can mix slowly |
| Diagnostics | Harder to diagnose convergence issues | Better diagnostics available |
| Implementation | Requires defining a guide | No guide needed |

The performance comparison from `test_pyro_comparison.py` shows that Pyro/SVI is typically faster than PyMC3/MCMC but may provide slightly different results due to its approximate nature.

## Observations in Pyro Models

Observations in Privugger's Pyro backend represent constraints on the model's outputs. These are critical for conditioning the model on known information, which is essential for privacy analysis.

### How Observations Are Represented

Observations are represented as string constraints such as:
- `"output >= 5"` (inequality)
- `"3 <= output <= 7"` (bounded)
- `"output == 10"` (equality)

These strings are parsed using a regex-based parser:

```python
def parse_observation(constraints_str):
    # Remove spaces for consistent parsing
    constraints = constraints_str.replace(" ", "")
    
    # Regular expression pattern for constraint parsing
    cons = r"[-+]?([0-9]*\.[0-9]+|[0-9]+)*([>=<]*)([a-zA-Z\s]*)([>=<]{2,})[-+]?([0-9]*\.[0-9]+|[0-9]+|\[(\d*,?)*\])*"
    vals = re.search(cons, constraints)
    
    # Extract components
    # ...
    
    return {
        "value1": v1,
        "constraint1": cons1,
        "name": name,
        "constraint2": cons2,
        "value2": v2
    }
```

### Adding Observations to Models

Observations are added to Pyro models using the `factor` operation, which modifies the log probability of the model:

```python
def add_pyro_observation(output, observation, precision):
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
        # Invert operator for proper conditioning
        op = obs["constraint1"].replace(">", "<")
        log_prob = apply_constraint(output, op, obs["value1"], precision)
        pyro.factor("obs_factor1", log_prob)
    
    # Apply second constraint if it exists
    if obs["value2"] is not None and obs["constraint2"]:
        log_prob = apply_constraint(output, obs["constraint2"], obs["value2"], precision)
        pyro.factor("obs_factor2", log_prob)
```

### Implementation of Soft Constraints

A key feature of the Pyro backend is its use of "soft constraints" that smoothly enforce the conditions:

```python
def apply_constraint(tensor, op, value, precision):
    if op == ">":
        # Use sigmoid to create soft constraint for tensor > value
        log_prob = -torch.sum(torch.nn.functional.relu(value - tensor)) / (2 * precision**2)
    elif op == ">=":
        log_prob = -torch.sum(torch.nn.functional.relu(value - tensor + 1e-6)) / (2 * precision**2)
    elif op == "<":
        log_prob = -torch.sum(torch.nn.functional.relu(tensor - value)) / (2 * precision**2)
    elif op == "<=":
        log_prob = -torch.sum(torch.nn.functional.relu(tensor - value + 1e-6)) / (2 * precision**2)
    elif op == "==":
        # Gaussian likelihood centered at value
        log_prob = -torch.sum(torch.pow(tensor - value, 2)) / (2 * precision**2)
    
    return log_prob
```

The use of soft constraints:
- Allows for continuous optimization in SVI
- Is controlled by a `precision` parameter (smaller values make constraints stricter)
- Avoids hard rejection sampling that would be computationally inefficient

### Impact of Observations on Inference

The tests in `test_pyro_observations.py` demonstrate how observations affect inference:

1. **Inequality constraints** (e.g., `"output >= 25"`) shift the posterior distribution toward the valid region
2. **Bounded constraints** (e.g., `"5 <= output <= 10"`) confine the posterior to the specified range
3. **Equality constraints** (e.g., `"output == 7"`) concentrate the posterior around the target value

The precision parameter controls how strictly these constraints are enforced:
- **Low precision** (e.g., 0.01): Stricter, more precise enforcement
- **High precision** (e.g., 2.0): More lenient, allowing more violation

## Integration with Privugger

The integration between the Pyro backend and the main Privugger framework happens in `privugger/inference/inference.py`:

```python
def infer(prog, ..., method="pymc3", ..., target_idx=0, svi_steps=1000, svi_lr=0.01):
    # ...
    elif method == "pyro":
        # Check if Pyro is available
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro backend is not available. Please install pyro-ppl")
        
        # Reset PyMC state if it was initialized
        if global_model_set:
            concatenated = False
            stacked = False
            global_model_set = False
            del global_model
            del global_priors
        
        return infer_pyro(
            prog, 
            input_specs, 
            output_type=output, 
            num_steps=svi_steps,
            num_samples=draws, 
            target_idx=target_idx,
            output_name=prog.name,
            chains=chains,
            lr=svi_lr
        )
```

This integration:
1. Exposes a unified API for both PyMC and Pyro backends
2. Configures SVI parameters through the main `infer` function
3. Handles conversion between different data formats

## Example Usage

Here's an example of using the Pyro backend from Privugger:

```python
# Define a simple prior distribution
age = pv.Normal("age", mu=30.0, std=5.0)

# Create dataset and program
ds = pv.Dataset(input_specs=[age])
prog = pv.Program("output", dataset=ds, output_type=pv.Float, function=program_identity)

# Add an observation constraint
prog.add_observation("output >= 25", precision=0.01)

# Run inference with Pyro backend
trace = pv.infer(prog, draws=1000, method="pyro", svi_steps=5000, svi_lr=0.005)
```

## Conclusion

Privugger's Pyro backend provides an efficient implementation of SVI for probabilistic programming tasks, with particular strengths in:

1. **Performance**: SVI is generally faster than MCMC methods, especially for complex models
2. **Flexibility**: The implementation supports various distribution types and observation constraints
3. **Integration**: Seamless integration with the rest of Privugger's framework

The backend's approach to handling observations through soft constraints is particularly noteworthy, allowing for effective conditioning of models without the computational cost of rejection sampling.

For privacy analysis tasks, the SVI implementation is generally recommended over MCMC when:
- Performance is a priority
- The model dimensionality is high
- Approximate posterior estimates are acceptable

The precision parameter provides a useful knob for controlling the strictness of observation constraints, allowing users to balance between strict enforcement and optimization stability.
