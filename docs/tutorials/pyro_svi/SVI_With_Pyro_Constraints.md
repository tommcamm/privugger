# SVI with Pyro in Privugger: Understanding Constraints

This document explains the implementation of Stochastic Variational Inference (SVI) using Pyro in the Privugger framework, with special focus on how constraints and observations are handled.

## Background: Privug Method and Privacy Analysis

In the Privug method, we use probabilistic programming to assess privacy risks by:

1. Modeling an attacker's knowledge about system inputs as prior distributions
2. Encoding the program's behavior as a probabilistic model
3. Adding observations/constraints to represent what the attacker knows about the output
4. Inferring posterior distributions to determine what the attacker can learn

The **observation mechanism** is especially important, as it models the attacker's knowledge about system outputs and shapes how the posteriors are inferred.

## SVI Implementation in Pyro

Stochastic Variational Inference (SVI) is a technique for approximating posterior distributions. Unlike MCMC methods, SVI tries to find a simpler distribution (the guide) that closely matches the true posterior. This involves:

1. Defining a model that encodes priors and program behavior
2. Creating a guide (the approximation of the posterior)
3. Optimizing the parameters of the guide to minimize the distance from the true posterior
4. Sampling from the optimized guide to obtain posterior samples

## Observations and Constraints

### Key Challenge

The fundamental challenge with observations in SVI is that constraints are "soft" - unlike MCMC where we can simply reject samples that don't meet criteria, in SVI we need to encode constraints in a way that guides the optimization toward distributions that satisfy them.

### Implementation Details

In our Pyro backend, constraints are handled through a combination of methods:

#### 1. Parsing Constraints

Observations are specified as strings like `"output == 7"` or `"5 <= output <= 10"` and parsed into components:

```python
# Example parsed constraint for "output >= 5"
{
  "name": "output",
  "constraint2": ">=",
  "value2": 5
}
```

#### 2. Applying Constraints in the Model

Constraints are applied using two primary mechanisms:

**a) Direct Observations (Equality Constraints)**

For equality constraints (`output == value`), we use a multi-layered approach:

```python
# 1. Primary observation with very small scale
pyro.sample(
    "obs_normal", 
    dist.Normal(output, effective_scale).to_event(output.dim()),
    obs=torch.tensor(value, dtype=output.dtype).expand_as(output)
)

# 2. Secondary observation with slightly different scale
pyro.sample(
    "obs_normal2", 
    dist.Normal(output, effective_scale * 2.0).to_event(output.dim()),
    obs=torch.tensor(value, dtype=output.dtype).expand_as(output)
)
```

This creates a strong likelihood spike at the target value. The precision parameter controls the scale - smaller precision means stricter enforcement.

**b) Factor Potentials (All Constraints)**

For all constraints (including equality), we also add factor potentials:

```python
# For equality:
log_prob = -0.5 * equality_scale * squared_diff

# For inequalities (e.g., >=):
violation = torch.nn.functional.softplus((value - tensor + 0.05) * sharpness) / sharpness
log_prob = -torch.sum(violation * base_scale_factor * penalty_multiplier)

# Add to model
pyro.factor("obs_factor", log_prob)
```

The factor adds an arbitrary log probability term to penalize violations. For inequalities, we use softplus for a smooth approximation that provides good gradients during optimization.

**c) Guide Shaping (Equality)**

For equality constraints, we also directly shape the guide:

```python
shape_param = pyro.param(
    "equality_shape", 
    torch.tensor(0.0, dtype=output.dtype),
    constraint=dist.constraints.real
)
pyro.factor(
    "equality_regularization",
    -10000.0 * torch.sum((shape_param - value)**2)
)
```

This creates a parameter the optimizer can't ignore, helping pull the guide toward solutions that satisfy the constraint.

#### 3. The Precision Parameter

The precision parameter controls the strictness of constraint enforcement:

- Smaller precision = stricter enforcement
- Larger precision = more relaxed enforcement

This is implemented through scaling factors:

```python
# More aggressive for small precisions
base_scale_factor = 100.0 / (effective_precision ** 2)
```

The precision transforms into penalty magnitudes - as precision approaches zero, penalties for violation approach infinity.

## SVI Optimization Process

The SVI process with constraints works as follows:

1. **Initialization**: Guide parameters are initialized based on priors
2. **Optimization**: The ELBO (Evidence Lower BOund) is maximized
   - Constraints add terms to the objective function
   - The optimizer tries to find guide parameters that satisfy constraints
3. **Sampling**: After optimization, we sample from the guide
4. **Post-processing**: Samples are converted to ArviZ format

## Key Implementation Insights

1. **Multiple Constraint Forms**: Using both `observe` and `factor` provides better guidance to the optimizer
2. **Smooth Approximations**: Using softplus for inequalities provides better gradients than hard cutoffs
3. **Precision Scaling**: The relationship between precision and scale factors is non-linear
4. **Guide Shaping**: Directly influencing the guide parameters helps optimization converge to better solutions

## Limitations and Considerations

- **Soft Constraints**: Even with strong penalties, constraints in SVI are inherently soft
- **Optimization Challenges**: Very strict constraints can make optimization difficult
- **Sample Efficiency**: Not all posterior samples will exactly satisfy constraints
- **Precision Tuning**: Finding the right precision value can require experimentation

## Usage in Privacy Analysis

In privacy analysis contexts:

1. The observations represent what an attacker knows about the output of a program
2. Precision represents how certain the attacker is about this knowledge
3. The posterior samples show what the attacker can infer about the input given this knowledge

By properly enforcing constraints, we get a more accurate picture of the privacy risks.
