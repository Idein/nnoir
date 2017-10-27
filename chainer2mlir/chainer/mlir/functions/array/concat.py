import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Concat(node.Function, functions.Concat):
    def __init__(self, *inputs, **dicts):
        super(Concat, self).__init__(functions.Concat)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {
                b'axis': self.axis
            }
        }

def concat(xs, axis=1):
    y, = Concat(axis).apply(xs)
    return y
