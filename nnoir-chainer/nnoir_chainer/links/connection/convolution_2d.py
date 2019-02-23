from chainer.links import Convolution2D
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import numpy as np
import nnoir.functions as NNOIR

Convolution2D.__call__ = patched_link_call(Convolution2D.__call__)


def to_nnoir_node(self, inputs, outputs):
    b = encode_ndarray(self.b.data) if (hasattr(self, 'b') and self.b is not None) else None
    is_depthwise = False
    out_channels, in_channels_per_groups, kh, kw = self.W.data.shape
    if self.groups > 1 and 1 == in_channels_per_groups:
        is_depthwise = True
    if is_depthwise:
        return NNOIR.DepthwiseConvolution2D(
            [x.name for x in inputs],
            [x.name for x in outputs],
            W=encode_ndarray(np.rollaxis(self.W.data.reshape(self.groups, out_channels//self.groups, kh, kw), 1, 0)),
            b=b,
            stride=self.stride,
            pad_h=(self.pad[0], self.pad[0]),
            pad_w=(self.pad[1], self.pad[1]),
            dilate=self.dilate
        )
    else:
        return NNOIR.Convolution2D(
            [x.name for x in inputs],
            [x.name for x in outputs],
            W=encode_ndarray(self.W.data),
            b=b,
            stride=self.stride,
            pad_h=(self.pad[0], self.pad[0]),
            pad_w=(self.pad[1], self.pad[1]),
            dilate=self.dilate,
            groups=self.groups
        )


Convolution2D.to_nnoir_node = to_nnoir_node
