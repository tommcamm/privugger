import numpy as np

def name(dataset: list[float],
                    max_value_dataset_entry: float,
                    min_value_dataset_entry: float,
                    delta: float,
                    epsilon: float) -> float:    
    avg   = sum(dataset)/len(dataset)
    sensitivity = (max_value_dataset_entry - min_value_dataset_entry)/len(dataset)
    noise_std = sensitivity*np.sqrt(2*np.log(1.25/delta))/epsilon
    noise = np.random.normal(loc=0,scale=noise_std)
    return avg + noise
