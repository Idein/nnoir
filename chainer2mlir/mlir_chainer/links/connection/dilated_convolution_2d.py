from chainer.links import DilatedConvolution2D
from chainer.mlir.patch import encode_ndarray, patched_link_call
import numpy as np

DilatedConvolution2D.__call__ = patched_link_call(DilatedConvolution2D.__call__)

def to_mlir_node(self):
    b = encode_ndarray(self.b.data) if (hasattr(self, 'b') and self.b is not None) else None
    return {
        b'name': 'Convolution2D',
        b'params': {
            b'W': encode_ndarray(self.W.data),
            b'b': b,
            b'stride': self.stride,
            b'pad_h' : (self.pad[0], self.pad[0]),
            b'pad_w' : (self.pad[1], self.pad[1]),
            b'dilate': self.dilate,
            b'groups': 1
        }
    }
DilatedConvolution2D.to_mlir_node = to_mlir_node
