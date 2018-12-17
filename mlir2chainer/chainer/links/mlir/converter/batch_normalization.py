import chainer.links as L
import numpy
import six

class ConvertBatchNormalization():

    def __init__(self, edge, inputs, outputs):
        self.bn = L.BatchNormalization(size = edge.params['avg_mean'].size,
                                       eps = edge.params['eps'],
                                       use_gamma = (edge.params['gamma'] is not None),
                                       use_beta = (edge.params['beta'] is not None),
                                       initial_gamma = edge.params['gamma'],
                                       initial_beta = edge.params['beta'],
                                       axis=None,
                                       initial_avg_mean = edge.params['avg_mean'],
                                       initial_avg_var = edge.params['avg_var'])

    def __call__(self, x):
        return self.bn(x)
