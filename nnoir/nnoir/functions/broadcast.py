from .function import Function
import numpy as np


class BroadcastTo(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'shape'}
        optional_params = set()
        super(BroadcastTo, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        return np.broadcast_to(x, self.params['shape'])
