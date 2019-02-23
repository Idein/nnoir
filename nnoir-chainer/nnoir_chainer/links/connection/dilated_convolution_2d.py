from chainer.links import DilatedConvolution2D
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import numpy as np
import nnoir.functions as NNOIR

DilatedConvolution2D.__call__ = patched_link_call(DilatedConvolution2D.__call__)


def to_nnoir_node(self, inputs, outputs):
    b = encode_ndarray(self.b.data) if (hasattr(self, 'b') and self.b is not None) else None
    return NNOIR.Convolution2D(
        [x.name for x in inputs],
        [x.name for x in outputs],
        W=encode_ndarray(self.W.data),
        b=b,
        stride=self.stride,
        pad_h=(self.pad[0], self.pad[0]),
        pad_w=(self.pad[1], self.pad[1]),
        dilate=self.dilate,
        groups=1
    )


DilatedConvolution2D.to_nnoir_node = to_nnoir_node
