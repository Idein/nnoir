from .function import Function
import numpy as np


class Gemm(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'transA', 'transB', 'alpha', 'beta'}
        optional_params = set()
        super(Gemm, self).__init__(inputs, outputs,
                                   params, required_params, optional_params)

    def run(self, a, b, c=None):
        transA = 0 if self.params['transA'] is None else self.params['transA']
        transB = 0 if self.params['transB'] is None else self.params['transB']
        a = a if transA == 0 else a.T
        b = b if transB == 0 else b.T
        result = np.dot(a, b)
        if self.params['alpha'] is not None:
            result *= self.params['alpha']
        if c is not None:
            if self.params['beta'] is not None:
                result += self.params['beta']*c
            else:
                result += c
        return result
