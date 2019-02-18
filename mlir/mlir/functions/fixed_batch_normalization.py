from .function import Function
import numpy as np


class FixedBatchNormalization(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'eps'}
        optional_params = set()
        super(FixedBatchNormalization, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x, gamma, beta, avg_mean, avg_var):
        shape = (1, gamma.size, 1, 1)
        gamma = gamma.reshape(shape)
        beta = beta.reshape(shape)
        avg_mean = avg_mean.reshape(shape)
        avg_var = avg_var.reshape(shape)
        return gamma * (x - avg_mean) / np.sqrt(avg_var + self.params['eps']) + beta
