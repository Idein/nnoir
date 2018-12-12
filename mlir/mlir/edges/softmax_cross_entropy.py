import chainer.functions as F
from .edge import Edge

class SoftmaxCrossEntropy(Edge):
    def __init__(self, inputs, outputs, **params):
        necessary_params = {'normalize',
                            'cache_score'}
        optional_params = set()
        super(SoftmaxCrossEntropy, self).__init__(inputs, outputs, params, necessary_params, optional_params)
