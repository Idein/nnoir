import chainer.functions as F
from .edge import Edge
import numpy as np

class ClippedReLU(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'upper'}
        optional_params = set()
        super(ClippedReLU, self).__init__(inputs, outputs, params, necessary_params, optional_params)
    def run(self, x):
        return np.maximum(0, np.minimum(x, self.params['upper']))
