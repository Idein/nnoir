import numpy as np
from .function import Function


class BatchNormalization(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'eps',
                           'avg_mean',
                           'avg_var',
                           'gamma',
                           'beta'}
        optional_params = set()
        super(BatchNormalization, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        shape = (1, self.params['gamma'].size) + ((1,) * (len(x.shape) - 2))
        gamma = self.params['gamma'].reshape(shape)
        beta = self.params['beta'].reshape(shape)
        avg_mean = self.params['avg_mean'].reshape(shape)
        avg_var = self.params['avg_var'].reshape(shape)
        return gamma * (x - avg_mean) / np.sqrt(avg_var + self.params['eps']) + beta
