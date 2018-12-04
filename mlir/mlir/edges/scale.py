import chainer.links as L
from .edge import Edge

class Scale(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'axis',
                            'W',
                            'bias.b'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
