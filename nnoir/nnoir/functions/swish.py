from .function import Function
import numpy as np


class Swish(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'beta'}
        optional_params = set()
        super(Swish, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        beta = self.params['beta']
        d = len(x.shape) - 1 - len(beta.shape)
        beta = beta.reshape((1,) + beta.shape + (1,)*d)
        return x * (0.5 * np.tanh(0.5 * beta * x) + 0.5)
