import chainer.functions as F
from .edge import Edge

class LeakyReLU(Edge):
    def __init__(self, inputs, outputs, **params):
        required_params = {'slope'}
        optional_params = set()
        super(LeakyReLU, self).__init__(inputs, outputs, params, required_params, optional_params)
    def run(self, x):
        R = x.copy()
        R[R < 0] *= self.params['slope']
        return R
