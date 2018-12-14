import numpy
import six
import chainer.links as L

class ConvertBias():
    def to_chainer(self, edge, *xs):
        bias = L.Bias(axis = edge.params['axis'],
                       shape = edge.params['b'].shape if edge.params['b'] is not None else None)
        bias.b.data = edge.params['b']
        return bias(*xs)
