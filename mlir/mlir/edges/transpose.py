import chainer.functions as F
from .edge import Edge

class Transpose(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'axes'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
    def run(self, x):
        return x.transpose(self.params['axes'])
