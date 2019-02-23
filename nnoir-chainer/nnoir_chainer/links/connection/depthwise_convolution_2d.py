from chainer.links import DepthwiseConvolution2D
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import nnoir.functions as NNOIR

DepthwiseConvolution2D.__call__ = patched_link_call(DepthwiseConvolution2D.__call__)


def to_nnoir_node(self, inputs, outputs):
    b = encode_ndarray(self.b.data) if (hasattr(self, 'b') and self.b is not None) else None
    return NNOIR.DepthwiseConvolution2D(
        [x.name for x in inputs],
        [x.name for x in outputs],
        W=encode_ndarray(self.W.data),
        b=b,
        stride=self.stride,
        pad_h=(self.pad[0], self.pad[0]),
        pad_w=(self.pad[1], self.pad[1]),
        dilate=(1, 1)
    )


DepthwiseConvolution2D.to_nnoir_node = to_nnoir_node
