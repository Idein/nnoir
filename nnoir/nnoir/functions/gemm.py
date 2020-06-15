from .function import Function
import numpy as np


class Gemm(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'transA', 'transB', 'alpha', 'beta'}
        optional_params = set()
        super(Gemm, self).__init__(inputs, outputs,
                                   params, required_params, optional_params)

    def run(self, a, b, c=None):
        a = a if self.params['transA'] == 0 else a.T
        b = b if self.params['transB'] == 0 else b.T
        result = self.params['alpha']*np.dot(a, b)
        if c is not None:
            result += self.params['beta']*c
        return result
