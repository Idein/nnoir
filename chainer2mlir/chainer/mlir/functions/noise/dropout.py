import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Dropout(node.Function, functions.Dropout):
    def __init__(self, *inputs, **dicts):
        super(Dropout, self).__init__(functions.Dropout)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {}
        }

def dropout(x, ratio=.5, **kwargs):
    return chainer.as_variable(x)
