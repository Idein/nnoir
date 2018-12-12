import chainer.links as L
import numpy
import six

class ConvertBatchNormalization():
    def to_chainer(edge, x):
        batch_norm = L.BatchNormalization(size = edge.params['avg_mean'].size,
                                          eps = edge.params['eps'],
                                          use_gamma = (edge.params['gamma'] is not None),
                                          use_beta = (edge.params['beta'] is not None))
        if edge.params['gamma'] is not None:
            batch_norm.gamma.data = edge.params['gamma']
        if edge.params['beta'] is not None:
            batch_norm.beta.data = edge.params['beta']
        batch_norm.avg_mean = edge.params['avg_mean']
        batch_norm.avg_var = edge.params['avg_var']
        return batch_norm(x)
