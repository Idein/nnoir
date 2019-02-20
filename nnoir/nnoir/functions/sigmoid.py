from .function import Function
import numpy as np


class Sigmoid(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = set()
        optional_params = set()
        super(Sigmoid, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return np.tanh(x * 0.5) * 0.5 + 0.5
