import chainer.functions as F
from .edge import Edge

class Dropout(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = set()
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
    def run(self, x):
        return x
