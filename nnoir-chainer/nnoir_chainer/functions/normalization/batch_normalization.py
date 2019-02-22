from chainer.functions.normalization.batch_normalization import FixedBatchNormalization
from nnoir_chainer.patch import patched_function_apply, patched_function_call

if hasattr(FixedBatchNormalization, 'apply'):
    FixedBatchNormalization.apply = patched_function_apply(FixedBatchNormalization.apply)
else:
    FixedBatchNormalization.__call__ = patched_function_call(FixedBatchNormalization.__call__)
