from .function import Function
import numpy as np


class Concat(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'axis'}
        optional_params = set()
        super(Concat, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, *xs):
        return np.concatenate(xs, self.params['axis'])
