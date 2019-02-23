from chainer.functions.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(SoftmaxCrossEntropy, 'apply'):
    SoftmaxCrossEntropy.apply = patched_function_apply(SoftmaxCrossEntropy.apply)
else:
    SoftmaxCrossEntropy.__call__ = patched_function_call(SoftmaxCrossEntropy.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.SoftmaxCrossEntropy(
        [x.name for x in inputs],
        [x.name for x in outputs],
        normalize=self.normalize,
        cache_score=self.cache_score,
    )


SoftmaxCrossEntropy.to_nnoir_node = to_nnoir_node
