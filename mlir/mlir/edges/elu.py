import chainer.functions as F
from .edge import Edge

class ELU(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'alpha'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
