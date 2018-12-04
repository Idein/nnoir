import chainer.functions as F
from .edge import Edge

class Mul(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = set()
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
