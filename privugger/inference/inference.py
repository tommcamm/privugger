
from sklearn import impute
from privugger.transformer.PyMC3.type_decoration import *
from privugger.distributions.continuous import Continuous
from privugger.distributions.discrete import Discrete, Constant, TensorConstant
from privugger.transformer.PyMC3.theano_types import TheanoToken
from privugger.transformer.PyMC3.program_output import *

import astor
import pymc as pm
import pytensor.tensor as at
import arviz as az
import os
import importlib
import warnings

# Import the Pyro backend
try:
    from privugger.inference.pyro import infer_pyro
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False

# Define which parameters are used by each backend
BACKEND_PARAMETERS = {
    "pymc3": {
        "required": [],
        "optional": ["cores", "chains", "draws", "return_model", "args_analyse", "args"]
    },
    "scipy": {
        "required": [],
        "optional": ["draws", "chains"]
    },
    "pyro": {
        "required": [],
        "optional": ["chains", "draws", "target_idx", "svi_steps", "svi_lr", 
                     "pyro_method", "autoguide", "guide_type", "warmup_steps"]
    }
}

# Common parameters used by all backends
COMMON_PARAMETERS = ["method", "prog"]

def warn_unused_parameters(method, provided_params, default_params=None):
    """
    Check for parameters that are not used by the specified backend method.
    Only warns about parameters that were explicitly provided (not defaults).
    
    Parameters
    ----------
    method : str
        The backend method name ("pymc3", "scipy", or "pyro")
    provided_params : dict
        Dictionary of parameter names and their values
    default_params : dict, optional
        Dictionary of default parameter values to compare against
        
    Returns
    -------
    None
        But issues warnings for unused parameters
    """
    # Get the parameters used by this backend
    if method not in BACKEND_PARAMETERS:
        return  # Unknown backend, can't check parameters
        
    backend_params = BACKEND_PARAMETERS[method]
    valid_params = set(COMMON_PARAMETERS + backend_params["required"] + backend_params["optional"])
    
    # Check for unused parameters (only those explicitly provided)
    unused_params = {}
    for param, value in provided_params.items():
        # Skip parameters that weren't explicitly provided (have default values)
        if default_params and param in default_params and value == default_params[param]:
            continue
            
        if param not in valid_params:
            unused_params[param] = value
    
    # Issue a single warning with all unused parameters
    if unused_params:
        # Format the unused parameters as a readable string
        params_str = ", ".join([f"'{p}' (value: {v})" for p, v in unused_params.items()])
        warnings.warn(
            f"The following parameters are not used by the '{method}' backend: {params_str}",
            UserWarning
        )

## Create a global pymc3 model and list of priors
global_model  = None
global_priors = []

## Set global variables for indicating if there was a concat/stack before infer
concatenated     = False
stacked          = False
global_model_set = False

def reset():
    global global_model, global_priors, concatenated, stacked, global_model_set
    global_model  = None
    global_priors = []
    concatenated     = False
    stacked          = False
    global_model_set = False

def _from_distributions_to_theano(input_specs, output):
    
    itypes = []
    otype = []
    
    
    if(input_specs == None):
        itypes.append(TheanoToken.float_matrix)
    else:
        for s in input_specs:
            if(isinstance(s, str)):
                if(s == "continuous"):
                    itypes.append(TheanoToken.float_vector)
                else:
                    itypes.append(TheanoToken.int_vector)

            elif(s.is_hyper_param):
                continue

            elif(issubclass(s.__class__, Continuous)):
                if(s.num_elements == -1):
                    itypes.append(TheanoToken.float_scalar)
                elif(s.num_elements==1):
                    itypes.append(TheanoToken.single_element_float_vector)
                else:
                    itypes.append(TheanoToken.float_vector)

            else:
                if(s.num_elements == -1):
                    itypes.append(TheanoToken.int_scalar)
                elif(s.num_elements==1):
                    itypes.append(TheanoToken.single_element_int_vector)
                else:
                    itypes.append(TheanoToken.int_vector)

    #NOTE: This gets the output type.
    if(type(output).__name__ ==  "type"):
        if(output.__name__ == "Float"):
           otype.append(TheanoToken.float_scalar)
        elif(output.__name__ == "Int"):
           otype.append(TheanoToken.int_scalar)
    elif(type(output).__name__ == "List"):
        if(output.output.__name__ == "Int"):  
           otype.append(TheanoToken.int_vector)
        elif(output.output.__name__ == "Float"): 
           otype.append(TheanoToken.float_vector)
    else:
        if(output.output.__name__ == "Int"):  
           otype.append(TheanoToken.int_matrix)
        elif(output.output.__name__ == "Float"): 
           otype.append(TheanoToken.float_matrix)

            

    return (itypes, otype)

def concatenate(distributions,  type_of_dist, axis=0):

    """
    
    Parameters
    ----------- 

    distribution_a: The first distribution
    
    distribution_b: The second distribution
    
    type_of_dist: String that specifies if it is a continuous or discrete distribution
    
    axis: Int value giving the axis to stack
     
    Returns
    -----------
    Type of distribution: String
    """
    
    #NOTE we just return a tuple and then actually concat later. First element is the distributions and second specify the axis and
    #if we are concatenating or stacking
    global global_model
    global global_model_set
    global global_priors
    
    if(not global_model_set):
        global_model     = pm.Model()
        global_priors    = []
        global_model_set = True
    with global_model as model:
        concatenated_variables = []
        for i in range(len(distributions)):
            concatenated_variables.append(distributions[i].pymc3_dist(distributions[i].name, []))
        global_priors.append(pm.math.concatenate(concatenated_variables, axis=0))
    global concatenated
    concatenated = True
    return type_of_dist
    #return ((distribution_a, distribution_b), (axis, "concat"))


def stack(distributions,  type_of_dist, axis=0):
    """
    
    Parameters
    -----------

    distributions: A list of distributions
     
    type_of_dist: String that specifies if it is a continuous or discrete distribution

    axis: Int value giving the axis to stack
     
    Returns
    -----------
    Type of distribution: String
    """
     
    #NOTE we just return a tuple and then actually stack later. First element is the distributions and second specify the axis and
    #if we are concatenating or stacking
    global global_model
    global global_model_set
    global global_priors
    
    if(not global_model_set):
        global_model     = pm.Model()
        global_priors    = []
        global_model_set = True
        
    with global_model as model:
        stacked_variables = []
        for i in range(len(distributions)):
            stacked_variables.append(distributions[i].pymc3_dist(distributions[i].name, []))
        global_priors.append(pm.math.stack(stacked_variables))

    global stacked
    stacked = True
    return type_of_dist
    #return (distributions, (axis, "stack"))


def get_model():
    if (not global_model_set):
        return None
    else:
        return global_model

def sample_prior(model, samples=50):

    """
    This method does the inference when provided a PyMC3 model

    Parameters
    -------------
    model : PyMC3 model
    
    samples : int number of samples

    Returns
    ------------
    
    Samples from the priors: Priors
    
    """
    with model as sample_prior:
        prior_checks = pm.sample_prior_predictive(samples=samples)

        return prior_checks
    
def infer(prog, cores=2, chains=2, draws=500, method="pymc3", return_model=False, args_analyse=3, args=None,
          target_idx=0, svi_steps=1000, svi_lr=0.01, suppress_param_warnings=False,
          pyro_method="svi", autoguide=False, guide_type="normal", warmup_steps=None):
    """
    Parameters
    -----------
    
    prog: the program type specified as a privugger.Program type
    
    cores: Int number of cores to use for sampling. Default 2
           Used by: 'pymc3' backend only
    
    chains: Int number of chains. Default 2
            Used by: All backends
    
    draws: Int number of draws. Default 500
           Used by: All backends

    method: String specifying which backend to use. Options: "pymc3", "scipy", "pyro"
            Default: "pymc3"

    return_model: Boolean. Returns the probabilistic model if true and the trace if false
                 Used by: 'pymc3' backend only

    args_analyse: Int default 3
                 Used by: 'pymc3' backend only

    args: Additional arguments (optional)
          Used by: 'pymc3' backend only

    target_idx: Int index of the target individual's distribution for Pyro SVI. Default 0
               Used by: 'pyro' backend only

    svi_steps: Int number of SVI steps for Pyro backend. Default 1000
              Used by: 'pyro' backend only
    
    svi_lr: Float learning rate for SVI optimizer in Pyro backend. Default 0.01
           Used by: 'pyro' backend only
    
    suppress_param_warnings: Boolean. If True, warnings about unused parameters will be suppressed. Default False
    
    pyro_method: String specifying the inference method for Pyro backend. Options: "svi", "mcmc", "hybrid"
                Default: "svi"
                - "svi": Run SVI only; return guide draws as idata.posterior
                - "hybrid": Run SVI then warm-started NUTS; guide's median seeds init_to_value
                - "mcmc": Skip SVI, run NUTS/HMC from scratch
                Used by: 'pyro' backend only
    
    autoguide: Boolean. If True, use Pyro's AutoGuide for variational inference. Default False
              Used by: 'pyro' backend only
    
    warmup_steps: Int number of warmup steps for MCMC in Pyro. Default None (uses num_steps // 5)
                 Used by: 'pyro' backend only

    Returns
    ----------
    Trace produced by the probabilistic programming inference: Arviz trace
    """
    # Collect all parameters to check for unused ones
    all_params = {
        "prog": prog, "cores": cores, "chains": chains, "draws": draws,
        "method": method, "return_model": return_model, "args_analyse": args_analyse,
        "args": args, "target_idx": target_idx, "svi_steps": svi_steps, "svi_lr": svi_lr,
        "pyro_method": pyro_method, "autoguide": autoguide, "guide_type": guide_type, 
        "warmup_steps": warmup_steps
    }
    
    # Define default parameter values
    default_params = {
        "cores": 2, "chains": 2, "draws": 500, "method": "pymc3", 
        "return_model": False, "args_analyse": 3, "args": None,
        "target_idx": 0, "svi_steps": 1000, "svi_lr": 0.01,
        "pyro_method": "svi", "autoguide": False, "guide_type": "normal", 
        "warmup_steps": None
    }
    
    # Warn about unused parameters if warnings aren't suppressed
    if not suppress_param_warnings:
        warn_unused_parameters(method, all_params, default_params)
    data_spec      = prog.dataset
    output         = prog.output_type
    num_specs      = len(data_spec.input_specs)
    input_specs    = data_spec.input_specs
    program        = prog.program

    global global_priors
    global global_model

    global concatenated
    global stacked
    global global_model_set
    
    if not (concatenated or stacked or global_model_set):
        
        global_model = pm.Model()
        global_priors = []
        global_model_set = True

    # global_model = pm.Model()
    # global_priors = []
    
    #### ##################
    ###### Lift program ###
    #######################
    if method == "pymc3":
        if(program is not  None):
            ftp = FunctionTypeDecorator()
            decorators = _from_distributions_to_theano(input_specs, output)
            lifted_program = ftp.lift(program, decorators)
            lifted_program_w_import = ftp.wrap_with_theano_import(lifted_program)
                
            f = open("typed.py", "w")
            f.write(astor.to_source(lifted_program_w_import))
            f.close()
        
                
            
            import typed as t
            importlib.reload(t)
            
            #################
            ## Create model #
            #################
            trace = None
            with global_model:
                
                priors = []
                hyper_params = []
                
                for idx in range(num_specs):
                    prior = input_specs[idx]
                    #This is for the case when our prior comes from a concatenated/stacked distribution
                    if(isinstance(prior, str)):
                        continue
                    if(prior.is_hyper_param):
                        hyper_params.append((prior, prior.name))
                    else:
                        params = prior.get_params()
                        hypers_for_prior = []
                        for p_idx in range(len(params)):
                            p = params[p_idx]
                            if(isinstance(p, Continuous) or isinstance(p, Discrete)):
                                for hyper in hyper_params:
                                    if(p.name == hyper[1]):
                                        hypers_for_prior.append((hyper[0],hyper[1], p_idx))
                            
                        #priors.append(prior.pymc3_dist(prior.name, hypers_for_prior))
                        global_priors.insert(idx, prior.pymc3_dist(prior.name, hypers_for_prior))
                if(program is not None):
                    #output = pm.Deterministic("output", t.method(*priors) )
                    output = pm.Deterministic(prog.name, t.method(*global_priors))
                # Add observations
                prog.execute_observations(prior, output)

                if(return_model):
                    return global_model
                else:
                    trace = pm.sample(draws=draws, chains=chains, cores=cores,return_inferencedata=True)

                concatenated     = False
                stacked          = False
                global_model_set = False
                del global_model
                del global_priors
                #global_model = pm.Model()
                return trace
            
    elif method == "scipy":
        if isinstance(program, str):
            import re
            f = open(program, "r")
            new = open("typed.py", "w")
            for l in f.readlines():
                res = re.findall(r"def [a-zA-Z]+\(", l)
                if len(res):
                    new.write(re.sub(r"def [a-zA-Z]+\(", "def method(", l))
                else:
                    new.write(l)
            f.close()
            new.close()
            import typed as t
            importlib.reload(t)
            f = t.method
        else:
            f = program
        priors = []
        trace = {}
        for idx in range(num_specs):
            name, dist = input_specs[idx].scipy_dist(input_specs[idx].name)
            dist = dist(draws)
            priors.append(dist)
            trace[name] = dist
        outputs = []
        for pi in list(zip(*priors)):
            if len(pi) == 1:
                pi = pi[0]
            outputs.append(f(*pi))
        trace["output"] = outputs
        return az.convert_to_inference_data(trace)
    
    elif method == "pyro":
        # Check if Pyro is available
        if not PYRO_AVAILABLE:
            raise ImportError("Pyro backend is not available. Please install pyro-ppl: pip install pyro-ppl")
        
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
            lr=svi_lr,
            method=pyro_method,
            autoguide=autoguide,
            guide_type=guide_type,
            warmup_steps=warmup_steps
        )
    
    else:
        raise TypeError(f"Unsupported probabilistic framework: {method}. Supported methods: 'pymc3', 'scipy', 'pyro'")
