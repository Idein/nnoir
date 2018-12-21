from chainer.functions.activation.relu import ReLU
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(ReLU, 'apply'):
    ReLU.apply = patched_function_apply(ReLU.apply)
else:
    ReLU.__call__ = patched_function_call(ReLU.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.ReLU(inputs, outputs)
ReLU.to_mlir_node = to_mlir_node
