from .function import Function
import numpy as np


class LocalResponseNormalization(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'n',
                           'k',
                           'alpha',
                           'beta'}
        optional_params = set()
        super(LocalResponseNormalization, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        RA2 = np.square(x)
        R = RA2.copy()
        for i in range(1, self.params['n']//2 + 1):
            R[:, i:] += RA2[:, :-i]
            R[:, :-i] += RA2[:, i:]
        R = self.params['k'] + self.params['alpha'] * R
        R = R ** -self.params['beta']
        R = x * R
        return R
