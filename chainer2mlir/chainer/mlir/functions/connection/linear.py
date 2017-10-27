import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class LinearFunction(node.Function, functions.LinearFunction):
    def __init__(self, *inputs, **dicts):
        super(LinearFunction, self).__init__(functions.LinearFunction)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {}
        }

def linear(x, W, b=None):
    if x.ndim > 2:
        x = x.reshape(len(x), -1)

    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction().apply(args)
    return y
