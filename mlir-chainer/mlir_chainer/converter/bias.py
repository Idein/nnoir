import numpy
import six
import chainer
import chainer.links as L

class ConvertBias(chainer.Chain):

    def __init__(self, edge, inputs, outputs):
        super(ConvertBias, self).__init__()
        with self.init_scope():
            self.bias = L.Bias(axis = edge.params['axis'],
                               shape = edge.params['b'].shape if edge.params['b'] is not None else None)
            self.bias.b.data = edge.params['b']

    def __call__(self, x):
        return self.bias(x)
