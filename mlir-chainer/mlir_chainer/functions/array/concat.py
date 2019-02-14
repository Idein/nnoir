from chainer.functions.array.concat import Concat
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(Concat, 'apply'):
    Concat.apply = patched_function_apply(Concat.apply)
else:
    Concat.__call__ = patched_function_call(Concat.__call__)


def to_mlir_node(self, inputs, outputs):
    return MLIR.Concat(inputs, outputs, axis=self.axis)


Concat.to_mlir_node = to_mlir_node
