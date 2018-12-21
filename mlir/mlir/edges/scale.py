import chainer.links as L
from .edge import Edge
import numpy as np

class Scale(Edge):
    def __init__(self, inputs, outputs, **params):
        required_params = {'axis',
                           'W',
                           'b'}
        optional_params = set()
        super(Scale, self).__init__(inputs, outputs, params, required_params, optional_params)
    def run(self, x):
        shape_post_len = len(x.shape) - self.params['axis'] - len(self.params['W'].shape)
        shape = (1,)*self.params['axis'] + self.params['W'].shape + (1,)*shape_post_len
        return x * np.reshape(self.params['W'], shape) + np.reshape(self.params['b'], shape)
