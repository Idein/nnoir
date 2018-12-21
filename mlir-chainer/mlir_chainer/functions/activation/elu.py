from chainer.functions.activation.elu import ELU
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(ELU, 'apply'):
    ELU.apply = patched_function_apply(ELU.apply)
else:
    ELU.__call__ = patched_function_call(ELU.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.ELU(inputs, outputs, alpha=self.alpha)
ELU.to_mlir_node = to_mlir_node
