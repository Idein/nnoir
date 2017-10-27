import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class AveragePooling2D(node.Function, functions.AveragePooling2D):
    def __init__(self, *inputs, **dicts):
        super(AveragePooling2D, self).__init__(functions.AveragePooling2D)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {
                b'kh': self.kh,
                b'kw': self.kw,
                b'sy': self.sy,
                b'sx': self.sx,
                b'ph': self.ph,
                b'pw': self.pw,
                b'cover_all': self.cover_all,
            }
        }

def average_pooling_2d(x, ksize, stride=None, pad=0):
    return AveragePooling2D(ksize, stride, pad, False).apply((x,))[0]
