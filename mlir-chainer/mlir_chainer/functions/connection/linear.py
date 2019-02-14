from chainer.functions.connection.linear import LinearFunction
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(LinearFunction, 'apply'):
    LinearFunction.apply = patched_function_apply(LinearFunction.apply)
else:
    LinearFunction.__call__ = patched_function_call(LinearFunction.__call__)


def to_mlir_node(self, inputs, outputs):
    return MLIR.LinearFunction(inputs, outputs)


LinearFunction.to_mlir_node = to_mlir_node
