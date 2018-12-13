import chainer.functions as F
from .edge import Edge

class LinearFunction(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = set()
        optional_params = set()
        super(LinearFunction, self).__init__(inputs, outputs, params, necessary_params, optional_params)
    def run(self, x, W, b):
        return x.dot(W.transpose()) + b
