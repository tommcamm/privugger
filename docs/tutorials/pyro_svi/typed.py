import pytensor
import pytensor.tensor as at
import numpy as np


@pytensor.compile.ops.as_op(itypes=[at.dvector], otypes=[at.dscalar])
def method(dataset):

    def dp_gaussian_average_wrapper(dataset):
        """Wrapper with inline implementation instead of calling another function"""
        max_value = 10.0
        min_value = 0.0
        delta = 0.01
        epsilon = 1.0
        avg = sum(dataset) / len(dataset)
        sensitivity = (max_value - min_value) / len(dataset)
        noise_std = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(loc=0, scale=noise_std)
        return np.array(avg + noise)
    return dp_gaussian_average_wrapper(dataset)
