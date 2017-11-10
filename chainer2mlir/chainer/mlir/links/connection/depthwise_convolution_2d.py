from chainer.links import DepthwiseConvolution2D
from chainer.mlir.patch import encode_ndarray, patched_link_call

DepthwiseConvolution2D.__call__ = patched_link_call(DepthwiseConvolution2D.__call__)

def to_mlir_node(self):
    b = encode_ndarray(self.b.data) if (hasattr(self, 'b') and self.b is not None) else None
    return {
        b'name': 'DepthwiseConvolution2D',
        b'params': {
            b'W': encode_ndarray(self.W.data),
            b'b': b,
            b'stride': self.stride,
            b'pad' : self.pad,
        }
    }
DepthwiseConvolution2D.to_mlir_node = to_mlir_node
