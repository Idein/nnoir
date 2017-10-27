import chainer
import chainer.links as links
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Convolution2D(node.Link, links.Convolution2D):
    def __init__(self, *inputs, **dicts):
        super(Convolution2D, self).__init__(links.Convolution2D)
        super(node.Link, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {
                b'W': encode_ndarray(self.W.data),
                b'b': encode_ndarray(self.b.data),
                b'stride': self.stride,
                b'pad' : self.pad,
            }
        }
