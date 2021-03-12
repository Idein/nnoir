from .function import Function
import numpy as np


class Linear(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'W',  # (out_ch, in_ch)
                           'b'}
        optional_params = set()
        super(Linear, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        result = x.reshape(x.shape[0], -1).dot(np.transpose(self.params['W']))
        if self.params['b'] is not None:
            result += self.params['b']
        return result
