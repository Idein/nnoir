from chainer.functions.normalization.batch_normalization import FixedBatchNormalization
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(FixedBatchNormalization, 'apply'):
    FixedBatchNormalization.apply = patched_function_apply(FixedBatchNormalization.apply)
else:
    FixedBatchNormalization.__call__ = patched_function_call(FixedBatchNormalization.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.FixedBatchNormalization(inputs, outputs)
FixedBatchNormalization.to_mlir_node = to_mlir_node
