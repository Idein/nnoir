from chainer.functions import LocalResponseNormalization
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(LocalResponseNormalization, 'apply'):
    LocalResponseNormalization.apply = patched_function_apply(LocalResponseNormalization.apply)
else:
    LocalResponseNormalization.__call__ = patched_function_call(LocalResponseNormalization.__call__)

def to_mlir_node(self):
    return {
        b'name': 'LocalResponseNormalization',
        b'params': {
            b'n': self.n,
            b'k': self.k,
            b'alpha': self.alpha,
            b'beta': self.beta
        }
    }
LocalResponseNormalization.to_mlir_node = to_mlir_node
