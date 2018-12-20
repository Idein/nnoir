from chainer.links import Convolution2D
from mlir_chainer.patch import encode_ndarray, patched_link_call
import numpy as np

Convolution2D.__call__ = patched_link_call(Convolution2D.__call__)

def to_mlir_node(self):
    b = encode_ndarray(self.b.data) if (hasattr(self, 'b') and self.b is not None) else None
    is_depthwise = False
    out_channels,in_channels_per_groups,kh,kw = self.W.data.shape
    if self.groups > 1 and 1 == in_channels_per_groups:
        is_depthwise = True
    if is_depthwise:
        return {
            b'name': 'DepthwiseConvolution2D',
            b'params': {
                b'W': encode_ndarray(np.rollaxis(self.W.data.reshape(self.groups,out_channels//self.groups,kh,kw),1,0)),
                b'b': b,
                b'stride': self.stride,
                b'pad_h' : (self.pad[0], self.pad[0]),
                b'pad_w' : (self.pad[1], self.pad[1]),
                b'dilate': self.dilate,
            }
        }
    else:
        return {
            b'name': 'Convolution2D',
            b'params': {
                b'W': encode_ndarray(self.W.data),
                b'b': b,
                b'stride': self.stride,
                b'pad_h' : (self.pad[0], self.pad[0]),
                b'pad_w' : (self.pad[1], self.pad[1]),
                b'dilate': self.dilate,
                b'groups': self.groups
            }
        }
Convolution2D.to_mlir_node = to_mlir_node
