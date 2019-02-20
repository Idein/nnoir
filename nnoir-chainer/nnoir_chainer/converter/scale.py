import numpy
import chainer
import chainer.links as L


class ConvertScale(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertScale, self).__init__()
        with self.init_scope():
            self.scale = L.Scale(axis=function.params['axis'],
                                 W_shape=tuple(function.params['W'].shape),
                                 bias_term=(function.params['b'] is not None),
                                 bias_shape=function.params['b'].shape if function.params['b'] is not None else None)
            self.scale.W.data = function.params['W']
            if function.params['b'] is not None:
                self.scale.bias.b.data = function.params['b']

    def __call__(self, x):
        return self.scale(x)
