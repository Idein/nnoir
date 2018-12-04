import chainer.links as L
from .edge import Edge

class BatchNormalization(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'eps',
                            'avg_mean',
                            'avg_var',
                            'gamma',
                            'beta'}
        optional_params = set()
        super().__init__(inputs, outputs, params, necessary_params, optional_params)
