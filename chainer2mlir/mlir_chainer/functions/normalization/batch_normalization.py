from chainer.functions.normalization.batch_normalization import FixedBatchNormalization
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(FixedBatchNormalization, 'apply'):
    FixedBatchNormalization.apply = patched_function_apply(FixedBatchNormalization.apply)
else:
    FixedBatchNormalization.__call__ = patched_function_call(FixedBatchNormalization.__call__)

def to_mlir_node(self):
    return {
        b'name': 'FixedBatchNormalization',
        b'params': {}
    }
FixedBatchNormalization.to_mlir_node = to_mlir_node
