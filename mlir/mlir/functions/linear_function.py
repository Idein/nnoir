import chainer.functions as F
from .function import Function


class LinearFunction(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = set()
        optional_params = set()
        super(LinearFunction, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x, W, b):
        return x.dot(W.transpose()) + b
