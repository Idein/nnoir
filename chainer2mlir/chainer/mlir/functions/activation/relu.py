import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class ReLU(node.Function, functions.ReLU):
    def __init__(self, *inputs, **dicts):
        super(ReLU, self).__init__(functions.ReLU)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {}
        }

def relu(x):
    y, = ReLU().apply((x,))
    return y
