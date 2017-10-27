import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Reshape(node.Function, functions.Reshape):
    def __init__(self, *inputs, **dicts):
        super(Reshape, self).__init__(functions.Reshape)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {
                b'shape': self.shape
            }
        }

def reshape(x, shape):
    if x.shape == shape:
        return chainer.as_variable(x)
    y, = Reshape(shape).apply((x,))
    return y
