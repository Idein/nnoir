import numpy
import chainer
import chainer.links as L


class ConvertBias(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertBias, self).__init__()
        with self.init_scope():
            self.bias = L.Bias(axis=function.params['axis'],
                               shape=function.params['b'].shape if function.params['b'] is not None else None)
            self.bias.b.data = function.params['b']

    def __call__(self, x):
        return self.bias(x)
