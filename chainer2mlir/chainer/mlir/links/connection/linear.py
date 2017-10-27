import chainer
import chainer.links as links
from chainer.links.connection import linear
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Linear(node.Link, links.Linear):
    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(Linear, self).__init__(links.Linear)
        super(node.Link, self).__init__(in_size, out_size, nobias, initialW, initial_bias)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {
                b'W': encode_ndarray(self.W.data),
                b'b': encode_ndarray(self.b.data)
            }
        }
