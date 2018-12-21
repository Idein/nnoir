from chainer.links import Scale
from mlir_chainer.patch import encode_ndarray, patched_link_call
import mlir.edges as MLIR

Scale.__call__ = patched_link_call(Scale.__call__)

def to_mlir_node(self, inputs, outputs):
    bias_axis = self.bias.axis if (hasattr(self, 'bias') and self.bias is not None) else None
    bias_b = encode_ndarray(self.bias.b.data) if (hasattr(self, 'bias') and self.bias is not None) else None
    return MLIR.Scale(
        inputs,
        outputs,
        axis=self.axis,
        W=encode_ndarray(self.W.data),
        b=bias_b,
    )
Scale.to_mlir_node = to_mlir_node
