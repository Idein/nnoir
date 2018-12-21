import numpy
import six
import chainer
import chainer.links as L

class ConvertScale(chainer.Chain):

    def __init__(self, edge, inputs, outputs):
        super(ConvertScale, self).__init__()
        with self.init_scope():
            self.scale = L.Scale(axis = edge.params['axis'],
                                 W_shape = tuple(edge.params['W'].shape),
                                 bias_term = (edge.params['b'] is not None),
                                 bias_shape = edge.params['b'].shape if edge.params['b'] is not None else None)
            self.scale.W.data = edge.params['W']
            if edge.params['b'] is not None:
                self.scale.bias.b.data = edge.params['b']

    def __call__(self, x):
        return self.scale(x)
