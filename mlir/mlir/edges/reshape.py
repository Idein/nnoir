import chainer.functions as F
from .edge import Edge

class Reshape(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'shape'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
