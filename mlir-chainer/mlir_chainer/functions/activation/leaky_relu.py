from chainer.functions.activation.leaky_relu import LeakyReLU
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(LeakyReLU, 'apply'):
    LeakyReLU.apply = patched_function_apply(LeakyReLU.apply)
else:
    LeakyReLU.__call__ = patched_function_call(LeakyReLU.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.LeakyReLU(inputs, outputs, slope=self.slope)
LeakyReLU.to_mlir_node = to_mlir_node
