import chainer.links as L
from .edge import Edge
import numpy as np

class Linear(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'W', #(out_ch, in_ch)
                            'b'}
        optional_params = set()
        super(Linear, self).__init__(inputs, outputs, params, necessary_params, optional_params)
    def run(self, x):
        return x.dot(np.transpose(self.params['W'])) + self.params['b']
