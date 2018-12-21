from chainer.functions.activation.clipped_relu import ClippedReLU
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(ClippedReLU, 'apply'):
    ClippedReLU.apply = patched_function_apply(ClippedReLU.apply)
else:
    ClippedReLU.__call__ = patched_function_call(ClippedReLU.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.ClippedReLU(inputs, outputs, upper=self.cap)
ClippedReLU.to_mlir_node = to_mlir_node
