from .function import Function
import numpy as np


class ELU(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'alpha'}
        optional_params = set()
        super(ELU, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        R = x.copy()
        v = R < 0
        R[v] = self.params['alpha'] * (np.exp(R[v]) - 1)
        return R
