import chainer.functions as F
from .edge import Edge
import numpy as np

class Concat(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'axis'}
        optional_params = set()
        super(Concat, self).__init__(inputs, outputs, params, necessary_params, optional_params)
    def run(self, *xs):
        return np.concatenate(xs, self.params['axis'])
