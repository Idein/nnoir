import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Softmax(node.Function, functions.Softmax):
    def __init__(self, *inputs, **dicts):
        super(Softmax, self).__init__(functions.Softmax)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {}
        }

def softmax(x, axis=1):
    return Softmax(axis=axis).apply((x,))[0]
