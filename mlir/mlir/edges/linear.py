import chainer.links as L
from .edge import Edge

class Linear(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'W',
                            'b'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
