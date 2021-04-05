import numpy as np

from .function import Function


class Sin(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = set()
        optional_params = set()
        super(Sin, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        R = x.copy()
        return np.sin(R)
