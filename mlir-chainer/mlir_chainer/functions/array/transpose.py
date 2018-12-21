from chainer.functions.array.transpose import Transpose
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(Transpose, 'apply'):
    Transpose.apply = patched_function_apply(Transpose.apply)
else:
    Transpose.__call__ = patched_function_call(Transpose.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.Transpose(inputs, outputs, axes=self.axes)
Transpose.to_mlir_node = to_mlir_node
