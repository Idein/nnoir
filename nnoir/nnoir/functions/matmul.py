import numpy as np

from .function import Function


class MatMul(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = set()
        optional_params = set()
        super(MatMul, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, a, b):
        return np.matmul(a, b)
