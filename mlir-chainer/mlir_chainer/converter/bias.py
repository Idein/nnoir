import numpy
import six
import chainer.links as L

class ConvertBias():

    def __init__(self, edge, inputs, outputs):
        self.bias = L.Bias(axis = edge.params['axis'],
                           shape = edge.params['b'].shape if edge.params['b'] is not None else None)
        self.bias.b.data = edge.params['b']

    def __call__(self, x):
        return self.bias(x)
