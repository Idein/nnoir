from .function import Function
import numpy as np


class ConstantPadding(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'pads',
                           'value'}
        optional_params = set()
        super(ConstantPadding, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return np.pad(x, self.params['pads'],
                      mode='constant', constant_values=(self.params['value'],))
