import numpy
import six
import chainer.links as L

class ConvertLinear():
    def to_chainer(edge, x):
        linear = L.Linear(in_size = edge.params['W'].shape[1],
                          out_size = edge.params['W'].shape[0],
                          nobias = (edge.params['b'] is None))
        linear.W.data = edge.params['W']
        if edge.params['b'] is not None:
            linear.b.data = edge.params['b']
        return linear(x)
