from chainer.functions.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(SoftmaxCrossEntropy, 'apply'):
    SoftmaxCrossEntropy.apply = patched_function_apply(SoftmaxCrossEntropy.apply)
else:
    SoftmaxCrossEntropy.__call__ = patched_function_call(SoftmaxCrossEntropy.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.SoftmaxCrossEntropy(
        inputs,
        outputs,
        normalize=self.normalize,
        cache_score=self.cache_score,
    )
SoftmaxCrossEntropy.to_mlir_node = to_mlir_node
