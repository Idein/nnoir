import chainer
import chainer.links as L


class ConvertSwish(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertSwish, self).__init__()
        with self.init_scope():
            self.f = L.Swish(beta_shape=function.params['beta'].shape, beta_init=function.params['beta'])

    def __call__(self, x):
        return self.f(x)
