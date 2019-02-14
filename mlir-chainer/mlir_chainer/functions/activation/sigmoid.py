from chainer.functions.activation.sigmoid import Sigmoid
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(Sigmoid, 'apply'):
    Sigmoid.apply = patched_function_apply(Sigmoid.apply)
else:
    Sigmoid.__call__ = patched_function_call(Sigmoid.__call__)


def to_mlir_node(self, inputs, outputs):
    return MLIR.Sigmoid(inputs, outputs)


Sigmoid.to_mlir_node = to_mlir_node
