from .function import Function
import numpy as np


class Softmax(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'axis'}
        optional_params = set()
        super(Softmax, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return np.exp(x) / np.sum(np.exp(x), self.params['axis'], keepdims=True)
