import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Convolution2DFunction(node.Function, functions.Convolution2DFunction):
    def __init__(self, *inputs, **dicts):
        super(Convolution2DFunction, self).__init__(functions.Convolution2DFunction)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {}
        }

def convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, **kwargs):
    fnode = Convolution2DFunction(stride, pad, cover_all)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = fnode.apply(args)
    return y
