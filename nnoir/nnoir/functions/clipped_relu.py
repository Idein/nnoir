from .function import Function
import numpy as np


class ClippedReLU(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'upper'}
        optional_params = set()
        super(ClippedReLU, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return np.maximum(0, np.minimum(x, self.params['upper']))
