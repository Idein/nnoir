from chainer.functions.activation.tanh import Tanh
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(Tanh, 'apply'):
    Tanh.apply = patched_function_apply(Tanh.apply)
else:
    Tanh.__call__ = patched_function_call(Tanh.__call__)


def to_mlir_node(self, inputs, outputs):
    return MLIR.Tanh(inputs, outputs)


Tanh.to_mlir_node = to_mlir_node
