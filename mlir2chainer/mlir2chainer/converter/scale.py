import numpy
import six
import chainer.links as L

class ConvertScale():

    def __init__(self, edge, inputs, outputs):
        self.scale = L.Scale(axis = edge.params['axis'],
                             W_shape = tuple(edge.params['W'].shape),
                             bias_term = (edge.params['bias.b'] is not None),
                             bias_shape = edge.params['bias.b'].shape if edge.params['bias.b'] is not None else None)
        self.scale.W.data = edge.params['W']
        if edge.params['bias.b'] is not None:
            self.scale.bias.b.data = edge.params['bias.b']

    def __call__(self, x):
        return self.scale(x)
