from chainer.links import Scale
from chainer.mlir.patch import encode_ndarray, patched_link_call

Scale.__call__ = patched_link_call(Scale.__call__)

def to_mlir_node(self):
    bias_axis = self.bias.axis if (hasattr(self, 'bias') and self.bias is not None) else None
    bias_b = encode_ndarray(self.bias.b.data) if (hasattr(self, 'bias') and self.bias is not None) else None
    return {
        b'name': 'Scale',
        b'params': {
            b'axis': self.axis,
            b'W': encode_ndarray(self.W.data),
            b'bias.b': bias_b,
        }
    }
Scale.to_mlir_node = to_mlir_node
