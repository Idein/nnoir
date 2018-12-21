from chainer.functions.normalization.local_response_normalization import LocalResponseNormalization
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.edges as MLIR

if hasattr(LocalResponseNormalization, 'apply'):
    LocalResponseNormalization.apply = patched_function_apply(LocalResponseNormalization.apply)
else:
    LocalResponseNormalization.__call__ = patched_function_call(LocalResponseNormalization.__call__)

def to_mlir_node(self, inputs, outputs):
    return MLIR.LocalResponseNormalization(
        inputs,
        outputs,
        n=self.n,
        k=self.k,
        alpha=self.alpha,
        beta=self.beta
    )
LocalResponseNormalization.to_mlir_node = to_mlir_node
