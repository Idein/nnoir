from chainer.functions.array.reshape import Reshape
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(Reshape, 'apply'):
    Reshape.apply = patched_function_apply(Reshape.apply)
else:
    Reshape.__call__ = patched_function_call(Reshape.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.Reshape(inputs, outputs, shape=self.shape)
Reshape.to_mlir_node = to_mlir_node
