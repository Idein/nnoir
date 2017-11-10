from chainer.functions import SoftmaxCrossEntropy
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(SoftmaxCrossEntropy, 'apply'):
    SoftmaxCrossEntropy.apply = patched_function_apply(SoftmaxCrossEntropy.apply)
else:
    SoftmaxCrossEntropy.__call__ = patched_function_call(SoftmaxCrossEntropy.__call__)

def to_mlir_node(self):
    return {
        b'name': 'SoftmaxCrossEntropy',
        b'params': {
            b'normalize': self.normalize,
            b'cache_score': self.cache_score,
        }
    }
SoftmaxCrossEntropy.to_mlir_node = to_mlir_node
