from chainer.functions.noise.dropout import Dropout
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(Dropout, 'apply'):
    Dropout.apply = patched_function_apply(Dropout.apply)
else:
    Dropout.__call__ = patched_function_call(Dropout.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.Dropout(inputs, outputs)
Dropout.to_mlir_node = to_mlir_node
