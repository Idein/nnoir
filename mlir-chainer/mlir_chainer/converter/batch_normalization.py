import chainer
import chainer.links as L
import numpy


class ConvertBatchNormalization(chainer.Chain):

    def __init__(self, function, inputs, outputs):
        super(ConvertBatchNormalization, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(size=function.params['avg_mean'].size,
                                           eps=function.params['eps'],
                                           use_gamma=(function.params['gamma'] is not None),
                                           use_beta=(function.params['beta'] is not None),
                                           initial_gamma=function.params['gamma'],
                                           initial_beta=function.params['beta'],
                                           axis=None,
                                           initial_avg_mean=function.params['avg_mean'],
                                           initial_avg_var=function.params['avg_var'])

    def __call__(self, x):
        return self.bn(x)
